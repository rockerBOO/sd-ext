# Drop keys

## Help

```bash
$ python sd-ext/drop_keys.py --help
```

```
usage: drop_keys.py [-h] [--output OUTPUT] [--keys KEYS [KEYS ...]] [--overwrite] model

        Drop keys from the model using a string match.

        python drop_keys.py /my/lora/file.safetensors --output /my/lora/file-dropped-to_v.safetensors --keys to_v ff_net
        Dropped: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v.alpha
        Dropped: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight
        Dropped: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight
        Dropped: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_v.alpha
        Dropped: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight
        Dropped: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight
        Dropped: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj.alpha
        Dropped: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight
        Dropped: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight
        Dropped: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_2.alpha
        ...
        Dropped keys: 192


positional arguments:
  model                 LoRA model to check the norms of

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Output file to this file
  --keys KEYS [KEYS ...]
                        Keys to drop
  --overwrite           WARNING overwrites original file. Overwrite the model with dropped keys version
```
