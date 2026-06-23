import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import spectral_demo as spectral_demo  # noqa: E402


MODEL_NAME = os.getenv("MODEL_NAME", "flux-dev")

if MODEL_NAME not in spectral_demo.MODELS:
    raise ValueError(f"Unsupported MODEL_NAME={MODEL_NAME!r}. Choices: {sorted(spectral_demo.MODELS)}")

if not torch.cuda.is_available():
    raise RuntimeError("This Space requires a CUDA GPU. Attach GPU hardware before launching.")

spectral_demo.MODEL = spectral_demo.MODELS[MODEL_NAME]
spectral_demo.MODEL_NAME = MODEL_NAME
spectral_demo.REPO = spectral_demo.MODEL["repo"]
spectral_demo._patch_gradio_schema_bug()
spectral_demo.PIPE = spectral_demo.load_pipe()
demo = spectral_demo.build_ui()


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_api=False,
        show_error=True,
    )
