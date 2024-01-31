# SD Extensions

<!--toc:start-->
- [SD Extensions](#sd-extensions)
  - [Analysis](#analysis)
  - [Pruning](#pruning)
  - [Extracting](#extracting)
  - [Convert](#convert)
  - [Inference](#inference)
  - [Metadata](#metadata)
  - [Install](#install)
    - [Install using venv virtual environment](#install-using-venv-virtual-environment)
  - [Usage](#usage)
<!--toc:end-->

Script's I use with Stable Diffusion

## Analysis

- debug_vae_from_images.py
- clip_score.py
- clip_arch.py
- clip_embeddings.py
- clip_image_embeddings.py
- clip_iqa.py
- image_similarity.py
- check_norms.py
- fid.py

## Pruning

- scale_norms.py
- drop_keys.py

## Extracting

- extract_lora_block.py

## Convert

- convert_safetensors.py
- batch_bundle_convert.py

## Inference

- sdv.py
- sdxl-turbo.py
- wuerstechen.py

## Metadata

- read-metadata.py

## Install

Recommended cloning into another repository to use their dependencies.

### Install using venv virtual environment

```bash
python -m venv venv
source ./venv/bin/activate # linux
call .\venv\Scripts\activate.bat # windows

# PyTorch. Get the version that works for your computer.
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

## Usage

Script's should be setup for accelerate so launch the scripts like

```
accelerate launch check_norms.py
```

For each script check the help menu for how to use.

```
python check_norms.py --help
```
