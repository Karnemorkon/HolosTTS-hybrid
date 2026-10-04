"""Gradio веб-інтерфейс: текст, пресет голосу, клонування, вербалізація цифр."""
import gradio as gr

from app import tts_pipeline as pipe
from app import verbalizer


def create_demo():
    voices = pipe.list_voices()
    with gr.Blocks(title="HolosTTS hybrid (ONNX INT8 CPU)") as demo:
        gr.Markdown("# HolosTTS - український TTS (hybrid: ONNX Runtime INT8 + PyTorch-енкодер)")

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
                synth_btn = gr.Button("Синтезувати", variant="primary")

        def on_encode(path):
            if not path:
                return None
            return pipe.encode_voice(path)

        def on_verbalize(text):
            return verbalizer.auto_verbalize(text or "")

        def on_synth(text, preset, cloned, auto):
            if auto:
                text = verbalizer.auto_verbalize(text or "")
            voice = cloned if cloned is not None else preset
            sr, data = pipe.synthesize(text, voice)
            return sr, data

        audio_prompt.change(on_encode, inputs=[audio_prompt], outputs=[custom_voice])
        vb_btn.click(on_verbalize, inputs=[input_text], outputs=[input_text])
        synth_btn.click(
            on_synth,
            inputs=[input_text, speaker, custom_voice, auto_vb],
            outputs=[output_audio],
        )
        demo.queue(max_size=15)
    return demo
