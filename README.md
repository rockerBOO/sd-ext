# SD Extensions

<!--toc:start-->

- [SD Extensions](#sd-extensions)
  - [Analysis](#analysis)
  - [Pruning](#pruning)
  - [Install](#install)
    - [Install using venv virtual environment](#install-using-venv-virtual-environment)
  - [Usage](#usage)
  <!--toc:end-->

Script's I use with Stable Diffusion

## Analysis

- clip_iqa.py
- clip_score.py
- fid.py
- check_norms.py
- debug_vae_from_images.py
- read-metadata.py

## Pruning

- scale_norms.py
- drop_keys.py

## Install

Recommended cloning into another repository to use their dependencies.

### Install using venv virtual environment

```bash
python -m venv venv
source ./venv/bin/activate # linux
call .\venv\bin\activate.bat # windows

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
