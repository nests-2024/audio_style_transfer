import gradio as gr

from models import timedomain, ulyanov

example_audios = [
    ["wavs/corpus/johntejada-1.wav",
     "wavs/target/beat-box-2.wav"],

    ["wavs/songs/imperial.mp3",
     "wavs/songs/usa.mp3"],

    ["wavs/birds/BR_ALAGOAS_FOLIAGE/BR_AL_XI_2007_xeno_01_LIMPO.mp3",
     "wavs/birds/MEX_ALTAMIRA_ORIOLE/MEX_Altamira_Oriole-ACelisM_01.mp3"],

    ["wavs/birds/MEX_ALTAMIRA_ORIOLE/MEX_Altamira_Oriole-ACelisM_01.mp3",
    "wavs/birds/BR_ALAGOAS_FOLIAGE/BR_AL_XI_2007_xeno_01_LIMPO.mp3"],
]


def run_timedomain(content_path, style_path, sr=44100):
    synth_sr, synth_audio = timedomain.run(
        content_path,
        style_path,
        output_fname=None,
        input_features=['mags', 'real', 'imag'],
        n_fft=2048,          # 512 to sr / 2. Higher is better quality but is slower.
        n_layers=3,          # 1 to 3. Higher is better quality but is slower.
        n_filters=4096,      # 512 - 4096. Higher is better quality but is slower.
        hop_length=256,      # 256 to n_fft / 2. The lower this value, the better the temporal resolution.
        alpha=0.0005,        # 0.0001 to 0.01. The higher this value, the more of the original "content" bleeds through.
        k_w=5,               # 3 to 5. The higher this value, the more complex the patterns it can synthesize.
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
    title="Timedomain Audio Style Transfer",
    description="Combine style and content from two different audio files",

    fn=run_timedomain,
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

demo.launch(show_api=False, server_name="0.0.0.0")
#demo.launch(show_api=False)
