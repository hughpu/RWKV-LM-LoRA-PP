"""prepare checkpoint file from pretrained rwkv model in order for memory efficiency initialization

author: hughpu@hotmail.com
"""

import os
import gc
# use cpu large ram to do this job
# os.environ['DS_ACCELERATOR'] = "cpu"

from argparse import ArgumentParser

import torch
import deepspeed

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)
    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer

    parser.add_argument("--my_sample_len", default=0, type=int)
    parser.add_argument("--my_ffn_shift", default=1, type=int)
    parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_qa_mask", default=0, type=int)
    parser.add_argument("--my_testing", default='', type=str)

    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_load", default="", type=str)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=32, type=float)
    parser.add_argument("--lora_dropout", default=0.01, type=float)
    parser.add_argument("--lora_parts", default="att,ln,time", type=str)

    parser.add_argument("--precision", default="bf16", type=str)
    
    args = parser.parse_args()
    
    os.environ["RWKV_JIT_ON"] = "1"
    if args.lora and args.grad_cp == 1:
        print('!!!!! LoRA Warning: Gradient Checkpointing requires JIT off, disabling it')
        os.environ["RWKV_JIT_ON"] = "0"
    os.environ["RWKV_T_MAX"] = str(args.ctx_len)

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision

    from src.model import RWKV, RWKVPipe, LORA_CONFIG

    load_dict = torch.load(args.load_model, map_location="cpu")

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = args.n_embd * 4

    # only train lora parameters
    if args.lora:
        assert args.lora_r > 0, "LoRA should have its `r` > 0"
        LORA_CONFIG["r"] = args.lora_r
        LORA_CONFIG["alpha"] = args.lora_alpha
        LORA_CONFIG["dropout"] = args.lora_dropout
        LORA_CONFIG["parts"] = set(str(args.lora_parts).split(','))
        enable_time_finetune = 'time' in LORA_CONFIG["parts"]
        enable_ln_finetune = 'ln' in LORA_CONFIG["parts"]

    model = RWKV(args)
    print("got plain rwkv model from args.")

    if args.lora:
        model.requires_grad_(False)
        for name, module in model.named_modules():
            # have to check param name since it may have been wrapped by torchscript
            if any(n.startswith("lora_") for n, _ in module.named_parameters()):
                print(f'  LoRA training module {name}')
                for pname, param in module.named_parameters():
                    param.requires_grad = 'lora_' in pname
            elif enable_ln_finetune and '.ln' in name:
                print(f'  LoRA additionally training module {name}')
                for param in module.parameters():
                    param.requires_grad = True
            elif enable_time_finetune and any(n.startswith("time") for n, _ in module.named_parameters()):
                for pname, param in module.named_parameters():
                    if pname.startswith("time"):
                        print(f'  LoRA additionally training parameter {pname}')
                        param.requires_grad = True

    if args.load_partial == 1:
        load_keys = load_dict.keys()
        for k in model.state_dict():
            if k not in load_keys:
                load_dict[k] = model.state_dict()[k]

    model.load_state_dict(load_dict, strict=(not args.lora))
    print("loaded pretrained parameters into rwkv model.")
    
    del load_dict
    gc.collect()
    
    deepspeed.init_distributed()
    print("init dstributed environment.")
    
    pipe_model = RWKVPipe(args, num_stages=1)
    pipe_model.load_state_from_rwkv(model)
    print("shifted model parameters from rwkv to its pipeline module.")
    
    del model
    gc.collect()

    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)
    
    pipe_model.save_state_dict(save_dir=args.proj_dir)
    print(f"successfuly saved pretrained rwkv in pipeline mode checkpoints under {args.proj_dir}.")
