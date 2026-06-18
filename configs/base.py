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

# Hard-negative-aware PartialFC sampling (opt-in, default off = original random sampling).
# When enabled, a fraction of each step's negative-class budget is filled with classes
# whose centers are closest to the positive classes in the batch (cached kNN over class
# centers) plus classes recently found to produce high "confusing" logits, instead of
# pure uniform random. The rest of the budget stays random to preserve global coverage.
config.hard_neg_mining = False
config.hard_neg_ratio = 0.2            # max fraction of the PFC sample budget reserved for hard negatives
config.hard_neg_topk = 50              # cached nearest-neighbor classes per class center
config.hard_neg_warmup_epoch = 10      # epochs of pure-random sampling before enabling mining
config.hard_neg_refresh_interval = 2000  # steps between neighbor-cache refresh for a given class
config.hard_neg_queue_size = 8192      # size of the dynamic "recently confused" class queue

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
