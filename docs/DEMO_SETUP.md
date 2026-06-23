# Demo setup

This repo has two separate demo surfaces:

1. A static GitHub Pages site for the research roadmap.
2. A GPU-backed Gradio app for the interactive spectral demo.

## 1. Run the interactive Gradio demo

### Prerequisites

1. Install `uv`.
2. Make sure the machine has a CUDA-capable NVIDIA GPU.
3. Authenticate with Hugging Face if the selected model requires gated access:

   ```bash
   huggingface-cli login
   ```

### Fast path

From the repo root:

```bash
uv run experiments/spectral_demo.py --model flux-dev
```

Then open:

```text
http://127.0.0.1:7860
```

### Useful variants

FLUX token/latent tabs:

```bash
uv run experiments/spectral_demo.py --model flux-dev
```

SD3.5 velocity tab:

```bash
uv run experiments/spectral_demo.py --model sd3.5-medium
```

Custom port:

```bash
uv run experiments/spectral_demo.py --model flux-dev --port 7861
```

Public temporary Gradio share link:

```bash
uv run experiments/spectral_demo.py --model flux-dev --share
```

### If you are running on a remote GPU machine

Start the app on the remote machine:

```bash
uv run experiments/spectral_demo.py --model flux-dev
```

From your local machine, tunnel the port:

```bash
ssh -L 7860:localhost:7860 <user>@<host>
```

Then open `http://127.0.0.1:7860` locally.

## 2. Publish the static roadmap as GitHub Pages

GitHub Pages serves from `/docs`, not `/docs/roadmap`, so this repo uses `docs/index.html`
as a redirect into the generated roadmap.

### One-time setup

1. Clone the repo.
2. Confirm `docs/index.html` exists.
3. Push `main` to GitHub.
4. Enable Pages from the `docs` folder on `main`.

GitHub CLI command:

```bash
gh api -X POST repos/<owner>/<repo>/pages \
  -f source[branch]=main \
  -f source[path]=/docs
```

If Pages already exists, update it:

```bash
gh api -X PUT repos/<owner>/<repo>/pages \
  -f source[branch]=main \
  -f source[path]=/docs
```

### Verify

Check Pages status:

```bash
gh api repos/<owner>/<repo>/pages
```

The site URL will be:

```text
https://<owner>.github.io/<repo>/
```

That entrypoint redirects to:

```text
https://<owner>.github.io/<repo>/roadmap/
```

## 3. Refresh the roadmap site after content changes

The roadmap HTML is generated from `experiments/roadmap_registry.py`.

1. Edit the registry.
2. Rebuild:

   ```bash
   python experiments/make_roadmap.py
   ```

3. Commit the regenerated files under `docs/roadmap/`.
4. Push to `main`.
5. GitHub Pages updates automatically.

## 4. Repeatable next-time checklist

1. `git clone <repo-url>`
2. `cd colorful-noise`
3. `huggingface-cli login`
4. `uv run experiments/spectral_demo.py --model flux-dev`
5. Open `http://127.0.0.1:7860`
6. For static docs updates: `python experiments/make_roadmap.py`
7. Commit and push
8. Confirm Pages with `gh api repos/<owner>/<repo>/pages`
