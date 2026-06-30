import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import mlflow

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss, AdaFaceLoss
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

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

def cfg_to_params(cfg):
    params = {}
    for k, v in vars(cfg).items():
        if isinstance(v, (int, float, str, bool)):
            params[k] = v
        else:
            params[k] = str(v)
    return params  

def main(args):
    # get config
    cfg = get_config(args.config)
    params = cfg_to_params(cfg)

    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    if local_rank == 0 and cfg.mlflow==True:
        print("Connecting to mlflow server")
        MLFLOW_URI = "http://mlflow.cyberspace.vn"

        mlflow.set_tracking_uri(uri=MLFLOW_URI)
        mlflow.set_registry_uri(uri= MLFLOW_URI)
        mlflow.set_experiment("Face_Embedding")
        print("Connected")
        run_to_resume_id = "0cbe2bc3bf414096a30573a327b20c02"

        print(f"Resuming MLflow Run: {run_to_resume_id}")
        run_ctx = mlflow.start_run(run_name = cfg.run_name, run_id = run_to_resume_id)

    else:
        import contextlib
        run_ctx = contextlib.nullcontext()

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    checkpoint_path = os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
    if os.path.exists(checkpoint_path):
        logging.info(f"Checkpoint detected at {checkpoint_path}")
        if local_rank == 0:
            print(f"Checkpoint detected at {checkpoint_path}")
        cfg.resume=True
    else:
        if local_rank == 0:
            print(f"Training from scratch")

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    if local_rank == 0:
        print("Loading Data")
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
        cfg.use_albumentations,
        cfg.num_rec_files,
        cfg.use_synthetic_data,
        cfg.use_public_data,
        save_images=False,
        save_dir='../Data/webface42_subset',
        num_save=1000,
    )

    #Save subset images
    if local_rank == 0:
        print(f"Train Loader batch size: {train_loader.batch_size}")
        print("Loading Data Done")
        for i, (images, labels) in enumerate(train_loader):
            if i * train_loader.batch_size >= 1000:
                break
    if local_rank == 0:
        print(f"Number of batch: {len(train_loader)}")
    if local_rank == 0:
        print("Building Model")
    if cfg.network == "vit_l_dinov3":
        if local_rank == 0:
            print("Using backbone DINOv3")
        backbone = get_model(
            cfg.network, 
            dropout = 0.0,
            fp16=cfg.fp16, #Not affect to ViT backbone
            num_features = cfg.embedding_size,
            pretrained_path = cfg.pretrained_path,
            freeze_backbone = cfg.freeze_backbone,
            use_projection = cfg.use_projection,
            ).cuda()
    else:
        if local_rank == 0:
            print("Using backbone VIT L Insightface")
        backbone = get_model(
            cfg.network, 
            dropout = 0.0,
            fp16=cfg.fp16, #Not affect to ViT backbone
            num_features = cfg.embedding_size,
            ).cuda()
    if local_rank == 0:
        print("Wrapping")
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    if local_rank == 0:
        print("Hookfp16")
    backbone.register_comm_hook(None, fp16_compress_hook)
    if local_rank == 0:
        print("Set to train mode")
    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    if local_rank == 0:
        print("Set static graph")
    backbone._set_static_graph()

    model_ema = None
    if cfg.use_ema==True:
        if local_rank == 0:
            print("Using EMA")
        ema_decay = cfg.ema_decay
        ema_warmup = cfg.ema_warmup
        ema_update_after_step = cfg.ema_update_after_step 
        if rank == 0:
            model_ema = ModelEmaV3(
                backbone.module,
                decay=ema_decay,
                use_warmup=ema_warmup,
                update_after_step=ema_update_after_step,
                device=None,
                foreach=True,
            )
        if rank == 0:
            logging.info(f"EM Config")
            logging.info(f"EMA Decay: {ema_decay}")
            logging.info(f"EMA WARMUP: {ema_warmup}")
            logging.info(f"EMA Updater after {ema_update_after_step} batches/steps")
    if local_rank == 0:
        print("Done Built")
        print("Init Loss")
    if cfg.loss == "arcface":
        if local_rank==0:
            print(f"Using loss function ArcFace config: {cfg.loss}")
        margin_loss = CombinedMarginLoss(
            64,
            cfg.margin_list[0],
            cfg.margin_list[1],
            cfg.margin_list[2],
            cfg.interclass_filtering_threshold
        )
    elif cfg.loss == "adaface":
        if local_rank==0:
            print(f"Using loss function AdaFace config: {cfg.loss}")
        margin_loss = AdaFaceLoss(
            64,
            m=cfg.m,
            h=cfg.h,
            t_alpha=cfg.t_alpha,
            interclass_filtering_threshold=cfg.interclass_filtering_threshold
        )       

    cfg.total_batch_size = cfg.batch_size * world_size
    steps_per_epoch = cfg.num_image // cfg.total_batch_size // cfg.gradient_acc
    cfg.warmup_step = steps_per_epoch * cfg.warmup_epoch
    cfg.total_step = steps_per_epoch * cfg.num_epoch

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])

        if model_ema is not None and "state_dict_ema" in dict_checkpoint:
            model_ema.module.load_state_dict(dict_checkpoint["state_dict_ema"])
            if rank == 0:
                logging.info("Loaded EMA State from ckpt")
        elif model_ema is not None:
            if rank == 0:
                logging.warning("EMA is enabaled but no EMA State found in ckpt")

        del dict_checkpoint

    # Warm-start from a previous run's weights but keep a FRESH epoch/optimizer/lr-scheduler,
    # so this run trains its own schedule from step 0 (plain continued training via
    # `cfg.init_checkpoint`). Skipped when `cfg.resume` (an interrupted run of THIS output
    # dir is being continued instead, with its own optimizer/lr/epoch state).
    warm_start_checkpoint = cfg.init_checkpoint
    if not cfg.resume and warm_start_checkpoint:
        if warm_start_checkpoint.endswith(".pt"):
            # single model.pt: backbone-only warm start, FC classifier stays freshly initialized
            state_dict_backbone = torch.load(warm_start_checkpoint, map_location="cpu")
            backbone.module.load_state_dict(state_dict_backbone)
            if rank == 0:
                logging.info(f"[init_checkpoint] warm-started backbone only from {warm_start_checkpoint}")
        else:
            init_path = os.path.join(warm_start_checkpoint, f"checkpoint_gpu_{rank}.pt")
            if not os.path.exists(init_path):
                raise FileNotFoundError(f"init checkpoint not found: {init_path}")
            dict_init = torch.load(init_path, map_location="cpu")
            backbone.module.load_state_dict(dict_init["state_dict_backbone"])
            incompat = module_partial_fc.load_state_dict(
                dict_init["state_dict_softmax_fc"], strict=False)
            if rank == 0:
                logging.info(f"[init_checkpoint] warm-started backbone+FC from {init_path}")
                logging.info(f"[init_checkpoint] missing keys: {list(incompat.missing_keys)}")
                logging.info(f"[init_checkpoint] unexpected keys: {list(incompat.unexpected_keys)}")
            del dict_init

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )

    total_step_logging = (cfg.num_image // (cfg.batch_size * world_size)) * cfg.num_epoch
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        # total_step=cfg.total_step,
        total_step= total_step_logging,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer,
        mlflow = cfg.mlflow,
    )

    loss_am = AverageMeter()
    amp = torch.amp.grad_scaler.GradScaler('cuda', growth_interval=100)
    print("Training Loop")

    with run_ctx:
#        if local_rank == 0:
 #           try:
  #              mlflow.log_params(params)
   #         except mlflow.exceptions.RestException:
    #            print("Param 'rec' already logged, skipping...")
     #       mlflow.log_artifact("/workspace/data/code/arcface_torch/configs/wf42m_pfc03_40epoch_64gpu_vit_l.py", artifact_path="configs")
      #      mlflow.log_artifact("/workspace/data/code/arcface_torch/configs/base.py", artifact_path="configs")
       #     mlflow.log_artifact("/workspace/data/code/arcface_torch/train_v2.py", artifact_path="train")
        #    mlflow.log_artifact("/workspace/data/code/arcface_torch/dataset.py", artifact_path="dataset")

        for epoch in range(start_epoch, cfg.num_epoch):
            if isinstance(train_loader, DataLoader):
                train_loader.sampler.set_epoch(epoch)

            if cfg.sample_rate_schedule:
                scheduled_rate = None
                for sched_epoch, sched_rate in cfg.sample_rate_schedule:
                    if epoch >= sched_epoch:
                        scheduled_rate = sched_rate
                if scheduled_rate is not None and scheduled_rate != module_partial_fc.sample_rate:
                    if local_rank == 0:
                        print(f"[sample_rate_schedule] epoch {epoch}: PartialFC sample_rate "
                              f"{module_partial_fc.sample_rate} -> {scheduled_rate}")
                    module_partial_fc.set_sample_rate(scheduled_rate)

            for _, (img, local_labels) in enumerate(train_loader):
                global_step += 1
                local_embeddings = backbone(img)
                loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

                if cfg.fp16:
                    loss = loss / cfg.gradient_acc
                    amp.scale(loss).backward()
                    if global_step % cfg.gradient_acc == 0:
                        amp.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                        amp.step(opt)
                        amp.update()
                        opt.zero_grad()
                        lr_scheduler.step()
                else:
                    loss = loss / cfg.gradient_acc
                    loss.backward()
                    if global_step % cfg.gradient_acc == 0:
                        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                        opt.step()
                        opt.zero_grad()
                        lr_scheduler.step()
                
                if model_ema is not None and global_step % cfg.gradient_acc == 0:
                    model_ema.update(backbone, step = global_step)

                with torch.no_grad():
                    loss_am.update(loss.item(), 1)
                    callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                    # if local_rank == 0 and rank == 0 and global_step > 0:
                    #     mlflow.log_metric("train_loss", loss_am.avg, step=global_step)
                    #     mlflow.log_metric("learning_rate", lr_scheduler.get_last_lr()[0], step=global_step)

                    if global_step % cfg.verbose == 0 and global_step > 0:
                        if model_ema is not None:
                            if rank == 0:
                                logging.info(f"Running valid with EMA")
                                callback_verification(global_step, model_ema.module)
                    else:
                        callback_verification(global_step, backbone)

            if cfg.save_all_states:
                print("Saving all states")
                checkpoint = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "state_dict_backbone": backbone.module.state_dict(),
                    "state_dict_softmax_fc": module_partial_fc.state_dict(),
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict()
                }
                if model_ema is not None:
                    checkpoint["state_dict_ema"] = model_ema.module.state_dict()

                torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

            if rank == 0 and cfg.save_epoch==True:
                print(f"Saving model weights at epoch {epoch}")
                path_module = os.path.join(cfg.output, f"model_epoch_{epoch}.pt")
                torch.save(backbone.module.state_dict(), path_module)

            if rank == 0:
                path_module = os.path.join(cfg.output, "model.pt")
                torch.save(backbone.module.state_dict(), path_module)

                if model_ema is not None:
                    path_module_ema = os.path.join(cfg.output, "model_ema.pt")
                    torch.save(model_ema.module.state_dict(), path_module_ema)
                    
            if cfg.dali:
                train_loader.reset()

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if model_ema is not None:
                path_module_ema = os.path.join(cfg.output, "model_ema.pt")
                torch.save(model_ema.module.state_dict(), path_module_ema)

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
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())

