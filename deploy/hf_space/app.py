import os
import sys
from pathlib import Path

import gradio as gr
import torch


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import spectral_demo as spectral_demo  # noqa: E402


MODEL_NAME = os.getenv("MODEL_NAME", "flux-dev")

if MODEL_NAME not in spectral_demo.MODELS:
    raise ValueError(f"Unsupported MODEL_NAME={MODEL_NAME!r}. Choices: {sorted(spectral_demo.MODELS)}")

if torch.cuda.is_available():
    spectral_demo.MODEL = spectral_demo.MODELS[MODEL_NAME]
    spectral_demo.MODEL_NAME = MODEL_NAME
    spectral_demo.REPO = spectral_demo.MODEL["repo"]
    spectral_demo._patch_gradio_schema_bug()
    spectral_demo.PIPE = spectral_demo.load_pipe()
    demo = spectral_demo.build_ui()
else:
    with gr.Blocks(title="colorful-noise spectral demo") as demo:
        gr.Markdown(
            "# colorful-noise spectral demo\n\n"
            "This Space is deployed correctly, but the current runtime has no CUDA GPU attached.\n\n"
            "To run the actual `experiments/spectral_demo.py` app, switch the Space hardware to a GPU tier "
            "or an eligible ZeroGPU plan, then restart the Space.\n\n"
            f"Configured model: `{MODEL_NAME}`"
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_api=False,
        show_error=True,
    )
