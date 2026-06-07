<h1 align="center">
  Colorful-Noise:  <br>
  Training-Free Low-Frequency Noise Manipulation <br>for Color-Based Conditional Image Generation <br> [SIGGRAPH 2026]
</h1>

<p align='center'>
<a href="https://nadavc220.github.io/colorful-noise/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://youtu.be/6HpK2lZfnU0"><img src="https://img.shields.io/static/v1?label=YouTube&message=Video&color=orange"></a>
<a href="https://arxiv.org/abs/2605.00548"><img src="https://img.shields.io/badge/arXiv-2605.00548-b31b1b.svg"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-Red?logo=pytorch"></a>
</p>


This is the official repository of the paper "Colorful-Noise: Training-Free Low-Frequency Noise Manipulation for Color-Based Conditional Image Generation" by Nadav Z. Cohen, Ofir Abramovich, and Ariel Shamir.

![teaser](assets/teaser.png)

<br>

# Updates
[2026 May] Colorful-Noise code is officially released!<BR>
[2026 April] Colorful-Noise is accepted to SIGGRAPH 2026!<BR>

### TODOs
- [ x ] Upload Masked Noise Code
- [ x ] Upload Wavelets Code
- [ -- ] Upload more hand-drawn sketches
- [ -- ] Upload Flux Code

<br>

# Environment Setup
We recommend creating a conda environment with the latest library versions. In case issues arise, the library versions used in our experiments are mentioned below.
```
conda create -n colorful-noise python=3.11
conda activate colorful-noise

pip install torch torchvision torchaudio
pip install -U diffusers
pip install transformers
pip install accelerate

In case you wish to use wavelets:
pip install pytorch_wavelets
pip install ptwt
```

<br>


# Usage
For Colorful-Noise with SDXL and FFT:
```
python sdxl_fft.py --input_image inputs/savana.png --prompt "A photo of giraffe and an elephant in the savanna" --alpha 0.015 --gamma 0.05
```

Masked Outputs - Colorful-Noise can ignore a given color from your mask. To apply masking use the <--ignore_color> param.<br>
Note that masked inputs are more sensitive as they contain regular noise, which is less strict than coloring the entire noise latent.
```
python sdxl_fft.py --input_image inputs/cat_orange.png --prompt "A photo of cat in the park" --alpha 0.015 --gamma 0.075 --ignore_color "(0, 0, 0)" --seed 10
```

For Colorful-Noise with SDXL and Wavelets:
```
python sdxl_wavelets.py --input_image inputs/savana.png --prompt "A photo of giraffe and an elephant in the savanna" --level 3 --gamma 0.05
```




# Bibtex
If you found this project helpful in your research, please consider citing our paper.
```
@misc{cohen2026colorfulnoise,
      title={Colorful-Noise: Training-Free Low-Frequency Noise Manipulation for Color-Based Conditional Image Generation}, 
      author={Nadav Z. Cohen and Ofir Abramovich and Ariel Shamir},
      year={2026},
      eprint={2605.00548},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      doi={https://doi.org/10.1145/3799902.3811104},
      url={https://arxiv.org/abs/2605.00548}, 
}
```
