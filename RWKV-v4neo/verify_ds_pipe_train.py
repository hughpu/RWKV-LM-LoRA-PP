########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# this is for verifying the results of different models and make sure they agree with each other

import os, sys, types
import numpy as np
import torch
import deepspeed
from deepspeed.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
np.set_printoptions(precision=4, suppress=True, linewidth=200)
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

os.environ['RWKV_FLOAT_MODE'] = 'bf16' # bf16 or fp32
os.environ['RWKV_RUN_DEVICE'] = 'cuda' # currently model_train requires CUDA
RUN_DEVICE = os.environ['RWKV_RUN_DEVICE']

TOKEN_MODE = 'pile'

if TOKEN_MODE == 'pile':
    WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
    PIPE_MODEL_DIR = '/home/hughpu/data/model/RWKV-4-Pile-3B-20220910-165-Pipe'
    MODEL_NAME = '/home/hughpu/data/model/RWKV-4-Pile-3B-20220910-165'
    n_layer = 40
    n_embd = 5120
    ctx_len = 4096
    UNKNOWN_CHAR = None

from src.utils import TOKENIZER
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
if TOKEN_MODE == 'pile':
    tokenizer.vocab_size = 50277

########################################################################################################

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_T_MAX"] = str(ctx_len)

from src.model import RWKV, RWKVPipe
if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
    torch.set_default_dtype(torch.float16)
elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
    torch.set_default_dtype(torch.bfloat16)

args = types.SimpleNamespace()
args.vocab_size = tokenizer.vocab_size
args.ctx_len = ctx_len
args.n_embd = n_embd
args.n_layer = n_layer
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0
rwkv_org = RWKV(args).to(RUN_DEVICE)

print('loading ' + MODEL_NAME)
rwkv_ckpt = torch.load(MODEL_NAME + '.pth', map_location='cpu')
rwkv_org.load_state_dict(rwkv_ckpt)

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_PORT"] = str(TORCH_DISTRIBUTED_DEFAULT_PORT)
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["LOCAL_RANK"] = "0"
deepspeed.init_distributed(auto_mpi_discovery=False)
rwkv_pipe = RWKVPipe(args, num_stages=1)
rwkv_pipe.load_state_dir(PIPE_MODEL_DIR)

########################################################################################################

print(f"\nVerifying {os.environ['RWKV_RUN_DEVICE']} {os.environ['RWKV_FLOAT_MODE']}")

# context = '\nIn a'
context = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'

if TOKEN_MODE == 'pile':
    ctx = tokenizer.tokenizer.encode(context)
print(f'input len {len(ctx)} data {ctx}')

########################################################################################################

with torch.no_grad():
    print('\nRWKV-Origin output')
    out = rwkv_org.forward(torch.tensor([ctx]).to(RUN_DEVICE))[0].detach().cpu().float().numpy()
    print(out, '\n')

    print('\nRWKV-Pipe output')
    out = rwkv_pipe.forward(torch.tensor([ctx]).to(RUN_DEVICE))[0].detach().cpu().float().numpy()
    print(out, '\n')