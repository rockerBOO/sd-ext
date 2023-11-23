from safetensors import safe_open
path = "/mnt/900/builds/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/boo.safetensors"

with safe_open(path, "pt") as file:
    print(file.metadata())
