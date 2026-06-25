"""Model loaders + runners for the baseline phase. Run in the uv env (diffusers 0.38).

One model is loaded at a time (group-by-model execution), run over all its jobs, then torn
down to free VRAM. All FLUX transformers load 4-bit NF4 (fits the A5000 comfortably).

Patterns reused from experiments/spectral_demo.py:
  * 4-bit quantized FluxTransformer2DModel
  * Redux  = FluxPriorReduxPipeline -> prompt/pooled embeds -> text-free FluxPipeline
  * IP-Adapter = XLabs-AI/flux-ip-adapter + separate CLIP-large vision encoder
"""
from __future__ import annotations
import gc
import torch
from PIL import Image

# NVML allocator-assert workaround for one-shot (non-Gradio) quantized loads (proof_runner.py).
# Guarded so this module also imports in the anaconda env (broken diffusers) for StyleID-only runs.
try:
    import diffusers.models.model_loading_utils as _mlu
    _mlu._caching_allocator_warmup = lambda *a, **k: None
except Exception:
    pass

FLUX_DEV = "black-forest-labs/FLUX.1-dev"
REDUX_PRIOR = "black-forest-labs/FLUX.1-Redux-dev"
KONTEXT = "black-forest-labs/FLUX.1-Kontext-dev"
_D = "cuda"


def _free():
    gc.collect(); torch.cuda.empty_cache()


def _quant_transformer(repo):
    from diffusers import FluxTransformer2DModel, BitsAndBytesConfig
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16)
    return FluxTransformer2DModel.from_pretrained(
        repo, subfolder="transformer", quantization_config=qc, torch_dtype=torch.bfloat16)


def _load(img, size=512):
    im = img if isinstance(img, Image.Image) else Image.open(img)
    return im.convert("RGB").resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------- FLUX img2img
class FluxImg2Img:
    name = "flux_img2img"

    def load(self):
        from diffusers import FluxImg2ImgPipeline
        tr = _quant_transformer(FLUX_DEV)
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            FLUX_DEV, transformer=tr, torch_dtype=torch.bfloat16)
        self.pipe.text_encoder.to(_D); self.pipe.text_encoder_2.to(_D); self.pipe.vae.to(_D)
        self.pipe.set_progress_bar_config(disable=True)
        return self

    @torch.no_grad()
    def run(self, content, prompt, seed=0, steps=28, guidance=2.5, size=512, strength=0.6, **kw):
        img = _load(content, size)
        return self.pipe(prompt=prompt or "a high quality photo", image=img, strength=strength,
                         height=size, width=size, guidance_scale=guidance,
                         num_inference_steps=steps,
                         generator=torch.Generator(_D).manual_seed(seed)).images[0]

    def teardown(self):
        del self.pipe; _free()


# ---------------------------------------------------------------- FLUX Redux
class FluxRedux:
    name = "flux_redux"

    def load(self):
        from diffusers import FluxPriorReduxPipeline, FluxPipeline
        self.prior = FluxPriorReduxPipeline.from_pretrained(REDUX_PRIOR, torch_dtype=torch.bfloat16).to(_D)
        tr = _quant_transformer(FLUX_DEV)
        self.pipe = FluxPipeline.from_pretrained(
            FLUX_DEV, transformer=tr, torch_dtype=torch.bfloat16,
            text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None)
        self.pipe.vae.to(_D)
        self.pipe.set_progress_bar_config(disable=True)
        return self

    @torch.no_grad()
    def run(self, content, prompt=None, style=None, seed=0, steps=28, guidance=2.5, size=512, **kw):
        # Redux conditions on a reference image. For style transfer the *style* image is the
        # reference; for plain editing the content image is the reference (variation baseline).
        ref = _load(style if style is not None else content, size)
        out = self.prior(ref)
        return self.pipe(prompt_embeds=out.prompt_embeds.to(_D),
                         pooled_prompt_embeds=out.pooled_prompt_embeds.to(_D),
                         height=size, width=size, guidance_scale=guidance,
                         num_inference_steps=steps,
                         generator=torch.Generator(_D).manual_seed(seed)).images[0]

    def teardown(self):
        del self.pipe, self.prior; _free()


# ---------------------------------------------------------------- FLUX IP-Adapter
class FluxIPAdapter:
    name = "flux_ipadapter"

    def load(self):
        from diffusers import FluxPipeline
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        tr = _quant_transformer(FLUX_DEV)
        self.pipe = FluxPipeline.from_pretrained(FLUX_DEV, transformer=tr, torch_dtype=torch.bfloat16)
        self.pipe.text_encoder.to(_D); self.pipe.text_encoder_2.to(_D); self.pipe.vae.to(_D)
        self.pipe.load_ip_adapter("XLabs-AI/flux-ip-adapter", weight_name="ip_adapter.safetensors",
                                  image_encoder_pretrained_model_name_or_path=None)
        enc_id = "openai/clip-vit-large-patch14"
        self.enc = CLIPVisionModelWithProjection.from_pretrained(enc_id).eval()
        self.proc = CLIPImageProcessor.from_pretrained(enc_id)
        self.pipe.set_progress_bar_config(disable=True)
        return self

    @torch.no_grad()
    def _embeds(self, image):
        px = self.proc(images=image, return_tensors="pt").pixel_values
        return self.enc(pixel_values=px).image_embeds[None, :].float().cpu()

    @torch.no_grad()
    def run(self, content, prompt=None, style=None, seed=0, steps=28, guidance=3.5, size=512,
            ip_scale=0.9, **kw):
        ref = _load(style if style is not None else content, size)
        self.pipe.set_ip_adapter_scale(ip_scale)
        emb = self._embeds(ref)
        # diffusers 0.38 always prepares negative IP embeds; image_encoder is None here so we
        # must hand it precomputed zeros (true_cfg off, so they are unused but required).
        neg = torch.zeros_like(emb)
        return self.pipe(prompt=prompt or "a high quality photograph",
                         ip_adapter_image_embeds=[emb], negative_ip_adapter_image_embeds=[neg],
                         height=size, width=size, guidance_scale=guidance, true_cfg_scale=1.0,
                         num_inference_steps=steps,
                         generator=torch.Generator(_D).manual_seed(seed)).images[0]

    def teardown(self):
        del self.pipe, self.enc; _free()


# ---------------------------------------------------------------- FLUX Kontext
class FluxKontext:
    name = "flux_kontext"

    def load(self):
        from diffusers import FluxKontextPipeline
        tr = _quant_transformer(KONTEXT)
        self.pipe = FluxKontextPipeline.from_pretrained(
            KONTEXT, transformer=tr, torch_dtype=torch.bfloat16)
        self.pipe.text_encoder.to(_D); self.pipe.text_encoder_2.to(_D); self.pipe.vae.to(_D)
        self.pipe.set_progress_bar_config(disable=True)
        return self

    @torch.no_grad()
    def run(self, content, prompt=None, instruction=None, seed=0, steps=28, guidance=2.5,
            size=512, **kw):
        img = _load(content, size)
        instr = instruction or prompt or "improve this image"
        return self.pipe(image=img, prompt=instr, height=size, width=size,
                         guidance_scale=guidance, num_inference_steps=steps,
                         generator=torch.Generator(_D).manual_seed(seed)).images[0]

    def teardown(self):
        del self.pipe; _free()


# ---------------------------------------------------------------- StyleID / VGG-Gram (training-free)
class StyleIDGram:
    """Gatys neural style transfer: optimize pixels to match VGG content + style Gram.
    Fully training-free (frozen ImageNet VGG-19); content image is a hard anchor so it is
    the natural low-leakage style-transfer baseline."""
    name = "styleid"

    def load(self):
        from torchvision.models import vgg19, VGG19_Weights
        w = VGG19_Weights.IMAGENET1K_V1
        self.vgg = vgg19(weights=w).features.to(_D).eval()
        for m in self.vgg.modules():           # inplace ReLU breaks autograd through activations
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=_D).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=_D).view(1, 3, 1, 1)
        self.c_layers = {21}                       # relu4_2 content
        self.s_layers = {0, 5, 10, 19, 28}         # style
        return self

    def _t(self, img, size):
        import numpy as np
        a = torch.from_numpy(np.asarray(_load(img, size)).copy()).float().permute(2, 0, 1)[None] / 255
        return a.to(_D)

    def _feats(self, x):
        x = (x - self.mean) / self.std
        cf, sf = {}, {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.c_layers: cf[i] = x
            if i in self.s_layers:
                b, c, h, w = x.shape
                f = x.reshape(c, h * w)
                sf[i] = (f @ f.t()) / (c * h * w)
            if i >= max(max(self.c_layers), max(self.s_layers)):
                break
        return cf, sf

    def run(self, content, style=None, size=512, style_weight=1e6, **kw):
        opt_steps = 300                         # fixed; independent of the FLUX `steps` arg
        c = self._t(content, size); s = self._t(style, size)
        with torch.no_grad():
            cf, _ = self._feats(c)
            _, sf = self._feats(s)
        img = c.clone().contiguous().requires_grad_(True)
        opt = torch.optim.Adam([img], lr=0.03)
        for _ in range(opt_steps):
            opt.zero_grad()
            cf2, sf2 = self._feats(img.clamp(0, 1))
            cl = sum(((cf2[k] - cf[k]) ** 2).mean() for k in cf)
            sl = sum(((sf2[k] - sf[k]) ** 2).mean() for k in sf)
            (cl + style_weight * sl).backward()
            opt.step()
        from PIL import Image as I
        import numpy as np
        arr = (img.clamp(0, 1)[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        return I.fromarray(arr)

    def teardown(self):
        del self.vgg; _free()


REGISTRY = {
    "flux_img2img": FluxImg2Img,
    "flux_redux": FluxRedux,
    "flux_ipadapter": FluxIPAdapter,
    "flux_kontext": FluxKontext,
    "styleid": StyleIDGram,
}
