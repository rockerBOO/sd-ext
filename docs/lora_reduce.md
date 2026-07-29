# lora_reduce.py

Produces reduced variants of a LoRA file using different methods. Each experiment writes a separate output file with metadata recording what was done. Run all experiments at once or select specific ones with `--experiments`.

## Usage

```bash
# Run all experiments
python lora_reduce.py model.safetensors output_dir/

# Run specific experiments only
python lora_reduce.py model.safetensors output_dir/ --experiments max-norm norm-selective
python lora_reduce.py model.safetensors output_dir/ --experiments max-norm --max-norm 40

# Tune thresholds
python lora_reduce.py model.safetensors output_dir/ --threshold 2.0 --prune-threshold 3.0
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `input` | required | Input LoRA safetensors file |
| `output_dir` | required | Directory to write output variants |
| `--experiments` | all | Space-separated list of experiment names to run (see below) |
| `--global-scales` | 0.75 0.5 | Scale factors for global alpha reduction |
| `--selective-scale` | 0.5 | Alpha scale factor for outlier layers only |
| `--threshold` | 1.5 | Outlier threshold as multiple of median effective scale |
| `--svd-keep-ratio` | 0.75 | Fraction of singular values to keep in SVD truncation |
| `--prune-threshold` | 2.0 | Layer pruning: remove layers above this × median |
| `--energy-threshold` | 0.01 | Energy pruning: drop SV components below this fraction |
| `--max-norm` | off | Max-norm cap: clip effective scale to this absolute value |
| `--base-model` | off | Base model path (required for spec-ratio and orth experiments) |
| `--spec-ratio-target` | 0.9 | Target spectral ratio for spec-ratio reduction |
| `--orth-vecs` | 4 | Number of base singular vectors to orthogonalize against |
| `--cpu` | off | Force CPU processing |

## Experiments

### Scale experiments (modify alpha, weights untouched)
| Name | Description |
|---|---|
| `global` | Scale alpha uniformly across all layers by each factor in `--global-scales` |
| `selective` | Scale alpha only on outlier layers (above `--threshold × median`) |

### Norm experiments (modify weights, preserve alpha)
Applies `sqrt(factor)` symmetrically to both up and down matrices. Preserves LoRA+ factorization balance.

| Name | Description |
|---|---|
| `norm-global` | Scale all layers so effective scale → current median |
| `norm-selective` | Cap outlier layers down to median; leave normal layers untouched |
| `max-norm` | Cap any layer above `--max-norm` to that value; no-op on layers already below |

`max-norm` is the recommended starting point for preparing a LoRA for continued training — it brings outliers under control while keeping all keys present and factorization intact.

### Spectral experiments
| Name | Description |
|---|---|
| `spectral` | Scale all layers so their largest singular value → median spectral norm |
| `spec-ratio` | Reduce layers where `spectral_norm(LoRA) / spectral_norm(base) > target` (requires `--base-model`) |
| `orth` | Orthogonalize LoRA's dominant directions against base model's top singular vectors (requires `--base-model`) |

### Rank experiments
| Name | Description |
|---|---|
| `svd` | Truncate rank to keep top-k singular value components (keeps `--svd-keep-ratio` fraction of energy) |
| `energy` | Adaptive per-layer rank reduction: drop components below `--energy-threshold` fraction of layer energy |
| `effective_rank` | Adaptive per-layer rank reduction to each layer's entropy-based effective rank (`exp(entropy(s²/Σs²))`) — parameter-free, no threshold to tune |

### Pruning experiments (remove keys entirely)
| Name | Description |
|---|---|
| `layer-prune` | Remove all keys for layers above `--prune-threshold × median` |
| `block-prune` | Remove all keys for a block type: double, single, double-txt, double-img |

### Combo experiments
| Name | Description |
|---|---|
| `svd-norm-selective` | SVD truncation → cap remaining outliers to median |
| `norm-selective-global` | Cap outliers to median → scale everything by 0.75 |
| `spec-ratio-svd` | Spec-ratio reduction → SVD truncation (requires `--base-model`) |
| `pruned-norm-selective` | Layer pruning → cap remaining outliers |
| `energy-norm-selective` | Energy pruning → cap remaining outliers |
| `pruned-energy` | Layer pruning → energy pruning |

## Output Files

Each experiment produces a file named `{stem}-{experiment}.safetensors` in the output directory, with metadata keys recording the method, parameters, and resulting effective scale statistics.

## Choosing a Method

**For inference only**: pruning variants are fine. `block-prune double-txt` removes the highest-magnitude layer family entirely with minimal file size.

**For continued training**: use `max-norm` with an absolute cap (e.g. 2× median or a specific value from `lora_spec_ratio.py` analysis). This keeps all keys present with reduced but intact structure, preserving factorization balance.

**For targeted reduction**: `norm-selective` caps only layers above `threshold × median`, leaving normal layers completely untouched.

**For rank reduction without tuning a threshold**: `effective_rank` picks each layer's target rank directly from its singular-value spectrum's entropy, with no `--energy-threshold`-style parameter to tune. Use `energy` instead if you want manual control over how aggressive the reduction is.

## Key Concepts

**Effective scale** = `norm(up @ down) * (alpha / rank)`

**Alpha scaling vs weight scaling**: alpha scaling changes the multiplier without touching the weight matrices (direction and rank structure preserved). Weight scaling changes the matrices while keeping alpha (factorization balance preserved). These are not equivalent — alpha scaling also affects how the LoRA interacts with dropout and other per-layer operations.

**Why `sqrt(factor)` for weight scaling**: `norm(scaled_up @ scaled_down) = factor * norm(up @ down)`, so applying `sqrt(factor)` to both matrices gives an exact `factor` change in effective scale while keeping the up/down ratio constant.
