"""Gradio веб-інтерфейс: текст, пресет голосу, клонування, вербалізація цифр."""
import gradio as gr

from app import tts_pipeline as pipe
from app import verbalizer


DISCLAIMER_MD = '---\n\n> ⚠️ **Відмова від відповідальності / Disclaimer**\n>\n> * **UA:** Цей проєкт є незалежною аматорською розробкою (Docker-обгорткою) і **не є офіційним продуктом** автора оригінальної нейромережі. Він жодним чином не афілійований, не спонсорується і не підтримується розробником [patriotyk](https://github.com/patriotyk). Усі права на оригінальні моделі, ваги та препроцес належать їхньому законному автору. Проєкт надається "як є" (as is), автор цієї обгортки не несе відповідальності за можливі збої чи використання сервісу.\n> * **EN:** This project is an independent, community-driven Docker wrapper and **is not an official product** of the original neural network creator. It is not affiliated with, endorsed, or sponsored by [patriotyk](https://github.com/patriotyk). All rights to the original models and weights belong to their respective owner. The software is provided "as is", without warranty of any kind.'


def create_demo():
    voices = pipe.list_voices()
    with gr.Blocks("HolosTTS hybrid (ONNX INT8 CPU)", delete_cache=(600, 1800)) as demo:
        gr.Markdown("# HolosTTS - український TTS (hybrid: ONNX Runtime INT8 + PyTorch-енкодер)")
        gr.Markdown(DISCLAIMER_MD)

        custom_voice = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Text(label="Текст", lines=6, max_lines=12)
                with gr.Row():
                    vb_btn = gr.Button("Вербалізувати цифри")
                    auto_vb = gr.Checkbox(
                        label="Автовербалізація цифр/дат/часу",
                        value=True,
                    )
                speaker = gr.Dropdown(
                    label="Голос (пресет)",
                    choices=voices,
                    value=voices[0] if voices else None,
                )
                audio_prompt = gr.Audio(
                    label="Аудіо для клонування голосу",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Результат", type="numpy", autoplay=False)
                with gr.Row():
                    save_name = gr.Textbox(
                        label="Імʼя нового голосу",
                        max_lines=1,
                        interactive=False,
                        info="Доступно після клонування (завантажте аудіо вище)",
                    )
                    save_btn = gr.Button(
                        "Зберегти голос",
                        variant="secondary",
                        interactive=False,
                    )
                save_status = gr.Markdown("Збереження голосу доступне лише після клонування.")
                synth_btn = gr.Button("Синтезувати", variant="primary")

        def on_encode(path):
            if not path:
                return None
            return pipe.encode_voice(path)

        def on_audio_change(path):
            # Клонування є → увімкнути збереження; немає → вимкнути.
            if not path:
                return None, gr.update(interactive=False), gr.update(interactive=False), "Збереження голосу доступне лише після клонування."
            emb = pipe.encode_voice(path)
            return (
                emb,
                gr.update(interactive=True),
                gr.update(interactive=True),
                "Клонування готове: вкажіть імʼя та натисніть «Зберегти голос».",
            )

        def on_verbalize(text):
            return verbalizer.auto_verbalize(text or "")

        def on_synth(text, preset, cloned, auto):
            if auto:
                text = verbalizer.auto_verbalize(text or "")
            voice = cloned if cloned is not None else preset
            sr, data = pipe.synthesize(text, voice)
            return sr, data

        def on_save(name, emb):
            if emb is None:
                return gr.update(), "Збереження можливе лише після клонування."
            exists = name in pipe.custom_voices()
            try:
                saved = pipe.save_voice(name, emb)
            except ValueError as e:
                return gr.update(), "Помилка: %s" % e
            all_voices = pipe.list_voices()
            note = " (перезаписано існуючий)" if exists else ""
            return (
                gr.update(choices=all_voices, value=saved),
                "Голос збережено%s: %s (доступний у API за цим імʼям)." % (note, saved),
            )

        save_btn.click(
            on_save,
            inputs=[save_name, custom_voice],
            outputs=[speaker, save_status],
        )
        audio_prompt.change(
            on_audio_change,
            inputs=[audio_prompt],
            outputs=[custom_voice, save_name, save_btn, save_status],
        )
        vb_btn.click(on_verbalize, inputs=[input_text], outputs=[input_text])
        synth_btn.click(
            on_synth,
            inputs=[input_text, speaker, custom_voice, auto_vb],
            outputs=[output_audio],
        )
        demo.queue(max_size=15)
    return demo
