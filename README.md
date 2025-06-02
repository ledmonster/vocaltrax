# VocalTrax

> [!NOTE]
> Code for the **NeurIPS Audio Imagination 2024** paper **[Articulatory Synthesis of Speech and Diverse Vocal Sounds via Optimization](https://openreview.net/forum?id=kIDxwvWznF)**. You can find examples [here](https://vocaltrax.media.mit.edu/).

Articulatory synthesis seeks to replicate the human voice by modeling the physics of the vocal apparatus, offering interpretable and controllable speech production. However, such methods often require careful hand-tuning to invert acoustic signals to their articulatory parameters. We present VocalTrax, a method which performs this inversion automatically via optimizing an accelerated vocal tract model implementation. Experiments on diverse vocal datasets show significant improvements over existing methods in out-of-domain speech reconstruction, while also revealing persistent challenges in matching natural voice quality.

> [!NOTE]
> For our baseline comparison, we refer you to [Vocal Tract Area Estimation by Gradient Descent](https://arxiv.org/abs/2307.04702) and their [code](https://github.com/dsuedholt/vocal-tract-grad).

## Installation

You can create the environment as follows

```bash
conda create -n vocaltrax python=3.10
conda activate vocaltrax
pip install -r requirements.txt
```

By default, we install JAX for CPU. You can find more details in the [JAX documentation](https://github.com/google/jax#installation) on using JAX with your accelerators. We also install [CREPE: A Convolutional Representation for Pitch Estimation](https://arxiv.org/abs/1802.06182), which requires tensorflow and it's installed for CPU by default.

## Running

You can resynthesize a target sound as follows

```bash
cd vocaltrax
python synthesize.py general.iters=1000 general.frame_length=1024 general.hop_length=1024 general.target=data/valentine.wav
```

## Configuration

> [!IMPORTANT]
> We use [Hydra](https://hydra.cc/) to configure everything. The configuration can be found in `vocaltrax/conf/config.yaml`, with specific sub-configs in sub-directories of `vocaltrax/conf/`.

The configs define all the parameters (e.g. spectrogram, optimizer, preloss). By default, these are the ones used for the paper. This is also where you choose the `target` file, `sample_rate`, `frame_length`, `hop_length`, `upsample_glottis`, number of iterations `iters`, learning rate `lr`, and the initial random `seed`.

## Acknowledgements & Citing

Please cite this work as follows:
```bibtex
@inproceedings{mo2024articulatory,
  title={Articulatory Synthesis of Speech and Diverse Vocal Sounds via Optimization},
  author={Mo*, Luke and Cherep*, Manuel and Singh*, Nikhil and Langford, Quinn and Maes, Patricia},
  booktitle={Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation}
}
```

MC received the support of a fellowship from “la Caixa” Foundation (ID 100010434). The fellowship code is LCF/BQ/EU23/12010079.

This codebase inherited a significant part from [Vocal Tract Area Estimation by Gradient Descent](https://github.com/dsuedholt/vocal-tract-grad), for which we are thankful. They, in turn, adapted code from other projects such as Pink Trombone, which we adapt transitively.
