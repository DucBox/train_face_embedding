import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from model_ema import ModelEmaV3

from torch.utils.flop_counter import FlopCounterMode

def get_flops(model, inp_shape):
    model.to(device)
    model.eval()
    inp = torch.randn(inp_shape).to(device)
    flop_counter = FlopCounterMode(mods = model, display = False)
    with flop_counter:
        model(inp)

    return flop_counter.get_total_flops()

def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)


    # print("Saved done ~1k images")
    print("Building Model")
    backbone = get_model(
        cfg.network, 
        dropout=0.0, 
        fp16=cfg.fp16, 
        num_features=cfg.embedding_size,
        pretrained_path = cfg.pretrained_path,
        freeze_backbone = cfg.freeze_backbone,
        use_projection = cfg.use_projection
        ).cuda()

    if not cfg.use_projection:
        cfg.embedding_size = 1024
        print(f"No projection head, using embedidng size: {cfg.embedding_size}")

    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"Tong so params la {total_params}")

    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Tong so trainable params: {trainable_params}")
    device = "cpu"
    flops = get_flops(backbone, (1,3,112,112))
    print(f"Tong so FLOPS: {flops}")


if __name__ == "__main__":
    #CUDA: 0-1 la A100 
    #CUDA: 2-3 la T4

    import torch
    if torch.cuda.is_available():
        print("CUDA Available")
    else:
        print("No CUDA")

    print("Number off CUDA GPUs: ", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running on {torch.cuda.get_device_name(device)}")
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")

    parser.add_argument("--config", type=str, help="py config file")

    main(parser.parse_args())