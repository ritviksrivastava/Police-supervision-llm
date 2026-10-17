"""Fold the LoRA adapter into a full copy of the base checkpoint, for serving with vLLM (or any loader).

  python 08_merge_adapter.py --adapter ../runs/qwen35-4b-sn/final --out ../runs/qwen35-4b-sn/merged

Works tensor by tensor on the original safetensors shards (CPU only), so the merged checkpoint keeps the base
model's exact architecture and file layout: W' = W + (alpha / r) * B @ A for every adapted projection.
"""
import argparse, glob, json, os, re, shutil

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base_dir = snapshot_download(args.base)
    cfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]

    # adapter key "base_model.model.model.layers.3.mlp.up_proj.lora_A.weight" -> module "layers.3.mlp.up_proj"
    lora = {}
    with safe_open(os.path.join(args.adapter, "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            m = re.match(r"base_model\.model\.model\.(layers\.\d+\..+)\.lora_([AB])\.weight$", k)
            if not m:
                raise SystemExit(f"unexpected adapter tensor {k}")
            lora.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k).float()

    os.makedirs(args.out, exist_ok=True)
    merged = 0
    for shard in sorted(glob.glob(os.path.join(base_dir, "*.safetensors"))):
        tensors = {}
        with safe_open(shard, "pt") as f:
            meta = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                m = re.match(r"model\.language_model\.(layers\.\d+\..+)\.weight$", name)
                if m and m.group(1) in lora:
                    ab = lora[m.group(1)]
                    t = (t.float() + scale * (ab["B"] @ ab["A"])).to(t.dtype)
                    merged += 1
                tensors[name] = t.contiguous()
        save_file(tensors, os.path.join(args.out, os.path.basename(shard)), metadata=meta)
    if merged != len(lora):
        raise SystemExit(f"merged {merged} of {len(lora)} adapted modules - checkpoint layout not as expected")

    for f in os.listdir(base_dir):          # config, tokenizer, chat template, processor files, index
        if not f.endswith(".safetensors"):
            shutil.copy(os.path.join(base_dir, f), os.path.join(args.out, f))
    print(f"merged {merged} adapted modules (scale {scale}) into {args.out}")


if __name__ == "__main__":
    main()
