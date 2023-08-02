########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import logging
from argparse import ArgumentParser

import deepspeed
from deepspeed import DeepSpeedConfig
from deepspeed import comm as dist
from deepspeed.comm.comm import configure
from deepspeed.runtime.config import ZeroStageEnum
from deepspeed.utils.logging import LoggerFactory, log_dist
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology, PipelineParallelGrid
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

import os
import torch

GLOBAL_RANK = dist.get_world_rank_from_launcher()
WORLD_SIZE = dist.get_world_size_from_launcher()
NUM_NODES = int(os.environ["CROSS_SIZE"])
NUM_DEVICES = int(os.environ["LOCAL_SIZE"])
LOG = LoggerFactory.create_logger(name="rwkv", level=logging.INFO)


class DeltaTorchCheckPointEngine(TorchCheckpointEngine):
    def __init__(
        self,
        config_params=None,
        lora=True,
        enable_time_finetune=True,
        enable_ln_finetune=True
    ):
        super().__init__(config_params)
        self._lora = lora
        self._enable_time_finetune = enable_time_finetune
        self._enable_ln_finetune = enable_ln_finetune 

    def save(self, state_dict: dict, path: str):
        is_layer_module = "layer" in path and "model_state" in path
        if is_layer_module and self._lora and isinstance(state_dict, dict):
            state_dict = {
                mod_path: param
                for mod_path, param in state_dict.items()
                if self.is_modpath_delta(mod_path)
            }
        return super().save(state_dict, path)
    
    def is_modpath_delta(self, mod_path):
        if "lora_" in mod_path:
            return True
        
        if self._enable_time_finetune and ".ln" in mod_path:
            return True
        
        if self._enable_ln_finetune and "time_" in mod_path:
            return True

        return False

def wrap_rank(info: str):
    return f"[RANK {GLOBAL_RANK}] {info}"


def get_args():
    parser = ArgumentParser(description='RWKV Pipeline Parallelism Training')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline_parallel_size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser.add_argument("--load_model", default="", type=str)  # pretrained splited checkpoints directory with multiple layer_*_model_*.pt under it
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size can not be 0, no auto support for better dataset efficiency

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer

    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--my_pile_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--my_pile_shift", default=-1, type=int)  # my special pile mode - text shift
    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)

    parser.add_argument("--my_img_version", default=0, type=str)
    parser.add_argument("--my_img_size", default=0, type=int)
    parser.add_argument("--my_img_bit", default=0, type=int)
    parser.add_argument("--my_img_clip", default='x', type=str)
    parser.add_argument("--my_img_clip_scale", default=1, type=float)
    parser.add_argument("--my_img_l1_scale", default=0, type=float)
    parser.add_argument("--my_img_encoder", default='x', type=str)
    # parser.add_argument("--my_img_noise_scale", default=0, type=float)
    parser.add_argument("--my_sample_len", default=0, type=int)
    parser.add_argument("--my_ffn_shift", default=1, type=int)
    parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_qa_mask", default=0, type=int)
    parser.add_argument("--my_testing", default='', type=str)

    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_load", default="", type=str) # checkpoints directory for lora training
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=32, type=float)
    parser.add_argument("--lora_dropout", default=0.01, type=float)
    parser.add_argument("--lora_parts", default="att,ln,time", type=str)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":


    ########################################################################################################
    #
    # example: train a simple L12-D768 RWKV on dummy data
    #
    # python train.py --load_model "" --wandb "" --proj_dir "out" \
    # --data_file "" --data_type "dummy" --vocab_size 0 \
    # --ctx_len 128 --epoch_steps 1000 --epoch_count 20 --epoch_begin 0 --epoch_save 10 \
    # --micro_bsz 16 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 \
    # --lr_init 6e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    # --accelerator gpu --devices 1 --precision bf16 --strategy ddp_find_unused_parameters_false --grad_cp 0

    # example: train a simple L6-D512 RWKV from scratch on enwik8
    #
    # python train.py --load_model "" --wandb "" --proj_dir "out" \
    # --data_file "../data/enwik8" --data_type "utf-8" --vocab_size 0 \
    # --ctx_len 512 --epoch_steps 5000 --epoch_count 500 --epoch_begin 0 --epoch_save 5 \
    # --micro_bsz 12 --n_layer 6 --n_embd 512 --pre_ffn 0 --head_qk 0 \
    # --lr_init 8e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    # --accelerator gpu --devices 1 --precision bf16 --strategy ddp_find_unused_parameters_false --grad_cp 0

    # example: fine-tune RWKV 1.5B using 8xA100 40G = 1.76it/s = 115k token/s, VRAM 37477M
    #
    # python train.py --load_model "/fsx/BlinkDL/CODE/FP16/out_1b2/all-8040.pth" --wandb "" --proj_dir "out" \
    # --data_file "../data/train.npy" --data_type "numpy" --vocab_size 50277 \
    # --ctx_len 1024 --epoch_steps 1000 --epoch_count 1000 --epoch_begin 0 --epoch_save 5 \
    # --micro_bsz 8 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
    # --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    # --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0

    # example: fine-tune RWKV 1.5B using 1 GPU fp16 (VRAM 16G) NOTE: fp16 might overflow
    #
    # python train.py --load_model "/fsx/BlinkDL/CODE/FP16/out_1b2/all-8040.pth" --wandb "" --proj_dir "out" \
    # --data_file "../data/train.npy" --data_type "numpy" --vocab_size 50277 \
    # --ctx_len 1024 --epoch_steps 200 --epoch_count 1000 --epoch_begin 0 --epoch_save 1 \
    # --micro_bsz 11 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
    # --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    # --accelerator gpu --devices 1 --precision fp16 --strategy deepspeed_stage_2_offload --grad_cp 1

    args = get_args()

    deepspeed.init_distributed(
        dist_backend=args.backend
    )
    LOG.info(
        wrap_rank(
            f"torch distributed env is initialized({dist.is_initialized()}) and available({dist.is_available()})"
        )
    )
    ppsize = args.pipeline_parallel_size
    assert WORLD_SIZE % ppsize == 0, f"pipeline parallelism {ppsize} and world size {WORLD_SIZE} are not match."
    topology = PipeDataParallelTopology(num_dp=WORLD_SIZE // ppsize, num_pp=ppsize)
    mpu = PipelineParallelGrid(topology=topology)
    deepspeed_config = DeepSpeedConfig(args.deepspeed_config, mpu=mpu)
    
    if GLOBAL_RANK == 0:
        LOG.info("########## work in progress ##########")

    ########################################################################################################

    import os, warnings, datetime
    import numpy as np
    import torch
    import random

    if args.seed >= 0:
        LOG.info(wrap_rank(f"########## GLOBAL SEED {args.seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n"))
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = deepspeed_config.steps_per_print
    args.max_epochs = -1  # continue forever
    
    optimizer_params = deepspeed_config.optimizer_params
    args.betas = tuple(optimizer_params["betas"])
    args.real_bsz = deepspeed_config.train_batch_size
    args.lr_init = args.lr_final = optimizer_params["lr"]
    args.adam_eps = optimizer_params["eps"]
    
    scheduler_params = deepspeed_config.scheduler_params
    args.warmup_steps = scheduler_params["warmup_num_steps"] if scheduler_params else 0

    args.accelerator = dist.get_accelerator()._name.upper()

    os.environ["RWKV_T_MAX"] = str(args.ctx_len)
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = args.n_embd * 4

    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"

    # TODO: ignore pile_stage temporarely
    if args.my_pile_stage > 0:
        magic_prime_bak = args.magic_prime
        if args.ctx_len == 1024:
            args.magic_prime = 324331313
            args.epoch_count = 8043
        elif args.ctx_len == 2048:
            args.magic_prime = 162165671
            args.epoch_count = 4021
        elif args.ctx_len == 4096:
            args.magic_prime = 81082817
            args.epoch_count = 2010
        if args.my_pile_shift < 0:
            if args.ctx_len == 1024:
                args.my_pile_shift = 0
            elif args.ctx_len == 2048:
                args.my_pile_shift = 512
            elif args.ctx_len == 4096:
                args.my_pile_shift = 768

        if magic_prime_bak > 0:
            args.magic_prime = magic_prime_bak

        args.epoch_steps = 40320 // args.real_bsz
        assert args.epoch_steps * args.real_bsz == 40320
        if args.my_pile_stage == 2:
            assert args.lr_final == args.lr_init
        if args.my_pile_stage >= 2:  # find latest saved model
            list_p = []
            for p in os.listdir(args.proj_dir):
                if p.startswith("rwkv") and p.endswith(".pth"):
                    p = ((p.split("-"))[1].split("."))[0]
                    if p == "init":
                        p = -1
                    else:
                        p = int(p)
                    list_p += [p]
            list_p.sort()
            max_p = list_p[-1]
            if len(list_p) > 1:
                args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                if args.my_pile_stage == 2:
                    args.warmup_steps = 10
                else:
                    args.warmup_steps = 30
            args.epoch_begin = max_p + 1

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len

    float_mode = 'fp16' if deepspeed_config.fp16_enabled else 'fp32'
    float_mode = 'bf16' if deepspeed_config.bfloat16_enabled else float_mode

    if GLOBAL_RANK == 0:
        LOG.info(
        f"""
############################################################################
#
# RWKV-4 {float_mode.upper()} on {NUM_NODES}x{NUM_DEVICES} {args.accelerator}, bsz {deepspeed_config.train_batch_size}, zero: {deepspeed_config.zero_optimization_stage} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
# LoRA = {f'enabled, {args.lora_r} r, {args.lora_alpha} alpha, {args.lora_dropout} dropout, on {args.lora_parts}' if args.lora else 'disabled'}
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
#
############################################################################
"""
    )
        LOG.info(str(vars(args)) + "\n")

    assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "wds_img", "uint16"]

    if args.lr_final == 0 or args.lr_init == 0:
        if GLOBAL_RANK == 0:
            LOG.info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    os.environ["RWKV_FLOAT_MODE"] = float_mode
    if float_mode == "fp32":
        for i in range(10):
            if GLOBAL_RANK == 0:
                LOG.info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if float_mode == "fp16":
        if GLOBAL_RANK == 0:
            LOG.info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")
        torch.set_default_dtype(torch.float16)
    if float_mode == "bf16":
        torch.set_default_dtype(torch.bfloat16)

    os.environ["RWKV_JIT_ON"] = "1"
    if deepspeed_config.zero_optimization_stage == ZeroStageEnum.weights:
        os.environ["RWKV_JIT_ON"] = "0"
    if args.lora and args.grad_cp == 1:
        LOG.warn(
            wrap_rank('!!!!! LoRA Warning: Gradient Checkpointing requires JIT off, disabling it')
        )
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if float_mode == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    ########################################################################################################

    from src.dataset import PipeDataset
    from src.model import RWKVPipe, LORA_CONFIG

    lora=args.lora
    enable_time_finetune=False
    enable_ln_finetune=False
    if args.lora:
        assert args.lora_r > 0, "LoRA should have its `r` > 0"
        LORA_CONFIG["r"] = args.lora_r
        LORA_CONFIG["alpha"] = args.lora_alpha
        LORA_CONFIG["dropout"] = args.lora_dropout
        LORA_CONFIG["parts"] = set(str(args.lora_parts).split(','))
        enable_time_finetune = 'time' in LORA_CONFIG["parts"]
        enable_ln_finetune = 'ln' in LORA_CONFIG["parts"]

    pipe_module = RWKVPipe(args, num_stages=args.pipeline_parallel_size)
    LOG.info(wrap_rank("got rwkv pipeline module."))

    need_to_load_data = pipe_module._grid.is_first_stage or pipe_module._grid.is_last_stage
    trainset = PipeDataset(args) if need_to_load_data else None
    LOG.info(wrap_rank("got dataset for training."))

    LOG.info(wrap_rank(f"########## Loading {args.load_model}... ##########"))
    try:
        pipe_module.load_state_dir(
            args.load_model,
            strict=(not args.lora),
            checkpoint_engine=TorchCheckpointEngine()
        )
    except:
        LOG.info(wrap_rank(f"Bad checkpoint {args.load_model}"))
        exit(1)

    # If using LoRA, the LoRA keys might be missing in the original model
    if os.path.isdir(args.lora_load):
        pipe_module.load_state_dir(
            args.lora_load,
            strict=False,
            checkpoint_engine=TorchCheckpointEngine()
        )
        LOG.info(wrap_rank(f"loaded pretrained checkpoint from {args.lora_load}"))

    # only train lora parameters
    if args.lora:
        pipe_module.requires_grad_(False)
        for name, module in pipe_module.named_modules():
            # have to check param name since it may have been wrapped by torchscript
            if any(n.startswith("lora_") for n, _ in module.named_parameters()):
                LOG.info(wrap_rank(f'  LoRA training module {name}'))
                for pname, param in module.named_parameters():
                    param.requires_grad = 'lora_' in pname
            elif enable_ln_finetune and '.ln' in name:
                LOG.info(wrap_rank(f'  LoRA additionally training module {name}'))
                for param in module.parameters():
                    param.requires_grad = True
            elif enable_time_finetune and any(n.startswith("time") for n, _ in module.named_parameters()):
                for pname, param in module.named_parameters():
                    if pname.startswith("time"):
                        LOG.info(wrap_rank(f'  LoRA additionally training parameter {pname}'))
                        param.requires_grad = True

    trainable_parameters = [p for p in pipe_module.parameters() if p.requires_grad]
    LOG.info(wrap_rank(f"trainable parameter size is {sum(p.size().numel() for p in trainable_parameters)}"))
    LOG.info(wrap_rank(f"preparing engine, optimizer, dataset, etc."))
    pipe_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipe_module,
        model_parameters=trainable_parameters,
        training_data=trainset
    )
    
    assert isinstance(pipe_engine, deepspeed.PipelineEngine), "initialized engine should be pipe_engine"
    LOG.info(wrap_rank(f"replace checkpoint engine with delta engine to save trainable parameters only."))
    pipe_engine.checkpoint_engine = DeltaTorchCheckPointEngine(
        lora=args.lora,
        enable_ln_finetune=enable_ln_finetune,
        enable_time_finetune=enable_time_finetune
    )
    
    train_batch_size = pipe_engine.train_batch_size()
    grad_acc_steps = pipe_engine.gradient_accumulation_steps()
    micro_batch_size = pipe_engine.train_micro_batch_size_per_gpu()
    
    # tuned the arguments since they are not required to config at the same time
    grad_acc_steps = max(grad_acc_steps, train_batch_size // micro_batch_size)
    micro_batch_size = min(micro_batch_size, train_batch_size // grad_acc_steps)
    if GLOBAL_RANK == 0:
        LOG.info(
        f"Train batch size: {train_batch_size}, Gradient accumulation steps: {grad_acc_steps}, Micro batch size, {micro_batch_size}."
    )
    
    # 3D parallelism information
    data_parallelism = pipe_engine.grid.data_parallel_size
    model_parallelism = pipe_engine.grid.model_parallel_size
    pipeline_parallelism = pipe_engine.grid.pipe_parallel_size
    if GLOBAL_RANK == 0:
        LOG.info(
            f"Data Parallelism: {data_parallelism}, Model Parallelism: {model_parallelism}, Pipeline Parallelism: {pipeline_parallelism}."
        )

    if (args.lr_init > 1e-4 or train_batch_size < 8):
        if 'I_KNOW_WHAT_IM_DOING' in os.environ:
            if GLOBAL_RANK == 0:
                LOG.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                LOG.info(f'  WARNING: you are using too large LR ({args.lr_init} > 1e-4) or too small global batch size ({WORLD_SIZE} * {micro_batch_size} * {grad_acc_steps} < 8)')
                LOG.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            if GLOBAL_RANK == 0:
                LOG.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                LOG.info(f'  ERROR: you are using too large LR ({args.lr_init} > 1e-4) or too small global batch size ({WORLD_SIZE} * {micro_batch_size} * {grad_acc_steps} < 8)')
                LOG.info(f'  Unless you are sure this is what you want, adjust them accordingly')
                LOG.info(f'  (to suppress this, set environment variable "I_KNOW_WHAT_IM_DOING")')
                LOG.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            exit(0)

    for name, param in pipe_module.state_dict().items():
        shape = param.shape
        shape = [i for i in shape if i != 1]
        if len(shape) > 1:
            print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {name}")
        else:
            print(f"{str(shape[0]).ljust(5)}       {name}")


    save_every_steps = 100
    if GLOBAL_RANK == 0:
        LOG.info("ready to train.")
    for step in range(args.steps):
        loss = pipe_engine.train_batch()
        if step % deepspeed_config.steps_per_print == 0:
            if GLOBAL_RANK == 0:
                LOG.info(f"training loss: {loss} at step: {step}")
        if step % save_every_steps == 0:
            pipe_engine.save_checkpoint(args.proj_dir)
