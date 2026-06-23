from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.save_all_states = False
config.output = "ms1mv3_arcface_r50"

config.embedding_size = 512

# Partial FC
config.sample_rate = 1
config.interclass_filtering_threshold = 0

# PartialFC sample_rate schedule (opt-in). List of [start_epoch, sample_rate] pairs,
# e.g. [[0, 0.3], [60, 1.0]] keeps the base sample_rate until epoch 60, then switches
# to full sampling (no PFC subsampling) for a fine-tune tail to expose every negative
# class each step. Leave as None to keep config.sample_rate fixed for the whole run.
config.sample_rate_schedule = None

# ===== Progressive Cluster Optimization (PCO) — LVFace, arXiv:2501.13420 =====
# Stage of the 3-stage progressive schedule. Run as separate jobs, warm-starting each stage
# from the previous stage's checkpoint via `config.pco_init_checkpoint`.
#   1 = Feature Alignment      : plain CosFace + NCS  (default == original behaviour)
#   2 = Centroid Stabilization : maintain feature-expectation bank e_i + two-anchor loss (Eq.11)
#   3 = Boundary Refinement    : same two-anchor loss (Eq.12), set sample_rate=1.0 (NCS off)
config.pco_stage = 1
config.pco_proto_m1 = 0.4   # margin on classifier-weight anchor (cosθ_i - m1)
config.pco_proto_m2 = 0.4   # margin on feature-expectation anchor (cosθ_i^e - m2)
config.pco_scale = 64.0     # feature scale s (paper: 64)
# Algorithm 1 freezes e during stage 3; set True to keep updating the bank in stage 3 too.
config.pco_update_center_stage3 = False
# Directory of the previous stage's output (containing checkpoint_gpu_{rank}.pt) to warm-start
# from. Loads backbone + classifier (+ bank if present) but resets epoch/optimizer/lr. None = off.
config.pco_init_checkpoint = None

# General-purpose version of the above, for plain continued training outside the PCO staging
# scheme - e.g. "ran 60 epochs, now continue 30 more with a new lr/warmup/num_epoch schedule"
# rather than resuming the exact same run state. Loads weights only (fresh epoch/optimizer/lr).
# Set to either:
#   - a directory containing checkpoint_gpu_{rank}.pt (preferred: also warm-starts the FC
#     classifier weights, not just the backbone - matters when num_classes is unchanged)
#   - a single model.pt file (backbone only; FC classifier initializes fresh)
# Ignored when config.resume is True (an interrupted run of THIS output dir takes priority).
config.init_checkpoint = None

config.fp16 = False
config.batch_size = 128

# For SGD 
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1

config.verbose = 2000
config.frequent = 10

# For Large Sacle Dataset, such as WebFace42M
config.dali = False 
config.dali_aug = False

# Gradient ACC
config.gradient_acc = 1

# setup seed
config.seed = 2048

# dataload numworkers
config.num_workers = 2

# WandB Logger
config.wandb_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
config.suffix_run_name = None
config.using_wandb = False
config.wandb_entity = "entity"
config.wandb_project = "project"
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False # resume wandb run: Only if the you wand t resume the last run that it was interrupted
