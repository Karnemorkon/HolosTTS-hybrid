'''Wyoming TTS-адаптер: проксі до HolosTTS-hybrid REST API.

Легкий окремий контейнер (без моделей). Спілкується з holos-hybrid через
/v1/audio/speech (response_format=pcm, stream=true) і віддає PCM16-чунки
клієнтам Wyoming (Home Assistant Assist).
'''
import asyncio
import logging
import os
import time

import httpx
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.ping import Ping, Pong
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
)

logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
log = logging.getLogger('holos-wyoming')

HOLOS_URL = os.environ.get('HOLOS_URL', 'http://holos-hybrid:8002').rstrip('/')
PORT = int(os.environ.get('WYOMING_PORT', '10200'))
SAMPLE_RATE = 24000
WIDTH = 2
CHANNELS = 1
VOICE_CACHE_TTL = 5.0

_ATTR = Attribution(name='Karnemorkon', url='https://github.com/Karnemorkon/HolosTTS-hybrid')
_client = None
_voices_cache = None


def _get_client():
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=HOLOS_URL,
            timeout=httpx.Timeout(600.0, connect=10.0),
        )
    return _client


async def _fetch_voices():
    global _voices_cache
    now = time.monotonic()
    if _voices_cache and (now - _voices_cache[0]) < VOICE_CACHE_TTL:
        return _voices_cache[1]
    r = await _get_client().get('/api/voices')
    r.raise_for_status()
    voices = list((r.json() or {}).get('voices') or [])
    _voices_cache = (now, voices)
    return voices


async def _default_voice():
    try:
        voices = await _fetch_voices()
        if voices:
            return voices[0]
    except Exception:
        log.exception('Не вдалося отримати список голосів')
    return 'Speaker_84'


class HolosHandler(AsyncEventHandler):
    def __init__(self, reader, writer):
        super().__init__(reader, writer)
        self._stream_voice = None
        self._stream_text = []

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self._describe()
            return True
        if Synthesize.is_type(event.type):
            synth = Synthesize.from_event(event)
            voice = synth.voice.name if synth.voice and synth.voice.name else None
            await self._synthesize(synth.text, voice)
            return True
        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._stream_voice = start.voice.name if start.voice and start.voice.name else None
            self._stream_text = []
            return True
        if SynthesizeChunk.is_type(event.type):
            self._stream_text.append(SynthesizeChunk.from_event(event).text)
            return True
        if SynthesizeStop.is_type(event.type):
            text = ''.join(self._stream_text)
            voice = self._stream_voice
            self._stream_voice = None
            self._stream_text = []
            await self._synthesize(text, voice)
            return True
        if Ping.is_type(event.type):
            await self.write_event(Pong(text=Ping.from_event(event).text).event())
            return True
        log.debug('Ігноруємо подію: %s', event.type)
        return True

    async def _describe(self):
        try:
            voices = await _fetch_voices()
        except Exception:
            log.exception('describe: помилка отримання голосів')
            voices = []
        tts_voices = [
            TtsVoice(
                name=v,
                attribution=_ATTR,
                installed=True,
                description=None,
                version=None,
                languages=['uk'],
            )
            for v in voices
        ]
        program = TtsProgram(
            name='holos-hybrid',
            attribution=_ATTR,
            installed=True,
            description='Ukrainian TTS (HolosTTS INT8 ONNX)',
            version='1.0',
            voices=tts_voices,
        )
        await self.write_event(Info(tts=[program]).event())

    async def _resolve_voice(self, voice_name):
        '''Валідація імені голосу; невідоме -> дефолт (щоб HA не отримував тишу).'''
        default = await _default_voice()
        if not voice_name:
            return default
        try:
            voices = await _fetch_voices()
        except Exception:
            return voice_name if voice_name else default
        if voice_name in voices:
            return voice_name
        log.warning('Невідомий голос %r, використовую дефолтний %r', voice_name, default)
        return default

    async def _synthesize(self, text, voice_name):
        text = (text or '').strip()
        if len(text) < 1:
            await self.write_event(AudioStop().event())
            return
        voice = await self._resolve_voice(voice_name)
        await self.write_event(
            AudioStart(rate=SAMPLE_RATE, width=WIDTH, channels=CHANNELS).event()
        )
        client = _get_client()
        try:
            async with client.stream(
                'POST',
                '/v1/audio/speech',
                json={
                    'input': text,
                    'voice': voice,
                    'response_format': 'pcm',
                    'stream': True,
                },
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    log.error('holos HTTP %d: %s', resp.status_code, body[:300])
                    raise RuntimeError('holos HTTP %d' % resp.status_code)
                async for raw in resp.aiter_bytes():
                    if raw:
                        await self.write_event(
                            AudioChunk(
                                rate=SAMPLE_RATE,
                                width=WIDTH,
                                channels=CHANNELS,
                                audio=raw,
                            ).event()
                        )
        except Exception:
            log.exception('Синтез невдалий')
        finally:
            try:
                await self.write_event(AudioStop().event())
            except Exception:
                pass


async def _run():
    server = AsyncServer.from_uri('tcp://0.0.0.0:%d' % PORT)
    log.info('Wyoming TTS: tcp://0.0.0.0:%d (backend: %s)', PORT, HOLOS_URL)

    def factory(reader, writer):
        return HolosHandler(reader, writer)

    await server.run(factory)


if __name__ == '__main__':
    asyncio.run(_run())
