# lokr_reduce.py

Shrinks LyCORIS LoKr-format LoRA files by SVD-truncating each module's full-size
`lokr_w2` tensor into a decomposed `lokr_w2_a`/`lokr_w2_b` pair. Unlike standard
LoRA (`lora_up`/`lora_down`) files, LoKr files are often stored with `lokr_w2` as a
full, undecomposed matrix — most of the file size is in `lokr_w2`, not the tiny
`lokr_w1` factor — so there's no existing low-rank structure to truncate until you
decompose `lokr_w2` yourself. That's what this script does.

Use `lora_reduce.py` instead for standard LoRA or PEFT-format files (`lora_up`/
`lora_down` or `lora_A`/`lora_B` keys).

## Usage

```bash
# Run both experiments (default)
python lokr_reduce.py model.safetensors output_dir/

# Run one experiment only
python lokr_reduce.py model.safetensors output_dir/ --experiments svd
python lokr_reduce.py model.safetensors output_dir/ --experiments effective_rank

# Tune the SVD keep ratio
python lokr_reduce.py model.safetensors output_dir/ --experiments svd --svd-keep-ratio 0.3
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `input` | required | Input LoKr safetensors file |
| `output_dir` | required | Directory to write output variants |
| `--experiments` | all | Space-separated list of experiment names to run: `svd`, `effective_rank` |
| `--svd-keep-ratio` | 0.5 | SVD truncation: fraction of each module's rank to keep |
| `--cpu` | off | Force CPU processing |

## Experiments

| Name | Description |
|---|---|
| `svd` | Truncate every module's `lokr_w2` to `--svd-keep-ratio` fraction of its rank |
| `effective_rank` | Adaptive per-module truncation to each `lokr_w2`'s entropy-based effective rank (`exp(entropy(s²/Σs²))`) — parameter-free, no ratio to tune |

## Eligibility

Only modules where both `lokr_w1` and `lokr_w2` are stored in full (undecomposed)
format, and `lokr_w2` is 2D (`Linear`-style, not a 4D conv kernel), are reduced.
Modules that are already decomposed (`lokr_w2_a`/`lokr_w2_b` present) or use a 4D
conv `lokr_w2` are left untouched and reported as skipped in the CLI output — this
is not an error, just out of scope for the current implementation.

## Why `alpha` gets rewritten

This is the detail most likely to surprise someone reading the code: whenever a
module's `lokr_w2` is decomposed, this script **overwrites** that module's `.alpha`
tensor to `float(r)`, where `r` is the new rank — regardless of what alpha value (if
any) was in the input file.

This is required for correctness, not a stylistic choice. Verified against
ComfyUI's LoKr loader (`comfy/weight_adapter/lokr.py`): when both `lokr_w1` and
`lokr_w2` are stored full, the loader never resolves a rank to divide by, so it
falls back to an effective scale of `1.0` and ignores the stored alpha value
entirely (observed training-time alpha values in the wild are things like
`9.9992e9` — clearly unused sentinels). Once `lokr_w2` is decomposed into
`lokr_w2_a`/`lokr_w2_b`, the loader instead computes
`alpha_effective = stored_alpha / lokr_w2_b.shape[0]`. Setting the stored alpha to
exactly `r` (`lokr_w2_b`'s row count) makes `alpha_effective = r / r = 1.0`,
preserving today's effective output up to the SVD truncation error itself. Leaving
the original (often huge, sentinel) alpha value in place would silently scale the
LoKr delta by roughly `alpha / r`, producing garbage output that looks like a
crash-free but visually broken LoRA.

This behavior was verified against ComfyUI's loader specifically. Other loaders
(kohya-ss inference, diffusers) were not checked and may compute the LoKr scale
differently — if you're loading a reduced file somewhere other than ComfyUI, verify
the alpha-scaling convention matches before trusting the output.

## Output Files

Each experiment produces a file named `{stem}-{experiment}.safetensors` in the
output directory (`{stem}-svd.safetensors`, `{stem}-effrank.safetensors`), with
metadata keys (`lokr_reduce.method`, `lokr_reduce.modules_reduced`, etc.) recording
what was done.
