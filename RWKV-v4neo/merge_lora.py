from collections import OrderedDict
from pathlib import Path
import os
import sys

from typing import Dict
import typing
import torch

if '-h' in sys.argv or '--help' in sys.argv:
    print(f'Usage: python3 {sys.argv[0]} [--use-gpu] <lora_alpha> <base_model_ckpt_dir> <lora_checkpoint_dir> <output_dir>')

if sys.argv[1] == '--use-gpu':
    device = 'cuda'
    lora_alpha, base_model_dir, lora_dir, output_dir = float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
else:
    device = 'cpu'
    lora_alpha, base_model_dir, lora_dir, output_dir = float(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]

Path(output_dir).mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    for f in os.listdir(base_model_dir):
        print("-------- processing layer checkpoint {f} --------")
        base_file = os.path.join(base_model_dir, f)
        lora_file = os.path.join(lora_dir, f)
        out_file = os.path.join(output_dir, f)

        w: Dict[str, torch.Tensor] = torch.load(base_file, map_location='cpu')
        # merge LoRA-only slim checkpoint into the main weights
        w_lora: Dict[str, torch.Tensor] = torch.load(lora_file, map_location='cpu')
        for k in w_lora.keys():
            w[k] = w_lora[k]
        output_w: typing.OrderedDict[str, torch.Tensor] = OrderedDict()
        # merge LoRA weights
        keys = list(w.keys())
        for k in keys:
            if k.endswith('.weight'):
                prefix = k[:-len('.weight')]
                lora_A = prefix + '.lora_A'
                lora_B = prefix + '.lora_B'
                if lora_A in keys:
                    assert lora_B in keys
                    print(f'merging {lora_A} and {lora_B} into {k}')
                    assert w[lora_B].shape[1] == w[lora_A].shape[0]
                    lora_r = w[lora_B].shape[1]
                    w[k] = w[k].to(device=device)
                    w[lora_A] = w[lora_A].to(device=device)
                    w[lora_B] = w[lora_B].to(device=device)
                    w[k] += w[lora_B] @ w[lora_A] * (lora_alpha / lora_r)
                    output_w[k] = w[k].to(device='cpu', copy=True)
                    del w[k]
                    del w[lora_A]
                    del w[lora_B]
                    continue

            if 'lora' not in k:
                print(f'retaining {k}')
                output_w[k] = w[k].clone()
                del w[k]

        torch.save(output_w, out_file)
