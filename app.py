import gradio as gr

from models import timedomain, ulyanov

example_audios = [
    ["outs/examples/johntejada-1.wav", "outs/examples/beat-box-2.wav"],
    ["outs/examples/imperial.mp3", "outs/examples/usa.mp3"]
]


def run_timedomain(content_path, style_path, sr=44100):
    synth_sr, synth_audio = timedomain.run(
        content_path,
        style_path,
        output_fname=None,
        n_fft=2048,          # 512 to sr / 2. Higher is better quality but is slower.
        n_layers=1,          # 1 to 3. Higher is better quality but is slower.
        n_filters=4096,      # 512 - 4096. Higher is better quality but is slower.
        hop_length=256,      # 256 to n_fft / 2. The lower this value, the better the temporal resolution.
        alpha=0.0005,        # 0.0001 to 0.01. The higher this value, the more of the original "content" bleeds through.
        k_w=3,               # 3 to 5. The higher this value, the more complex the patterns it can synthesize.
        iterations=300,      # 100 to 1000. Higher is better quality but is slower.
        stride=1,            # 1 to 3. Lower is better quality but is slower.
        sr=sr,
    )
    return synth_sr, synth_audio


def run_ulyanov(content_path, style_path, sr=44100):
    synth_sr, synth_audio = ulyanov.run(
        content_path,
        style_path,
        output_fname=None,
        alpha=0.001,
        iterations=128,
        phase_iterations=256,
        sr=sr,
    )
    return synth_sr, synth_audio


demo = gr.Interface(
    title="Audio Style Transfer",
    description="Combine style and content from two different audio files",

    fn=run_ulyanov,
    inputs=[
        gr.Audio(type="filepath", source="upload", label="Content"),
        gr.Audio(type="filepath", source="upload", label="Style")
    ],
    outputs=[
        gr.Audio(label="Output"),
    ],

    examples=example_audios,
    cache_examples=True,

    allow_flagging="never",
    analytics_enabled=None
)

#demo.launch(show_api=False, server_name="0.0.0.0")
demo.launch(show_api=False)
