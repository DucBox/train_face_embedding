from easydict import EasyDict as edict

# ===== PCO Stage 3: Boundary Refinement (LVFace, arXiv:2501.13420) =====
# Warm-starts from the Stage-2 run. Same two-anchor loss (Eq.12) but NCS is DISABLED
# (sample_rate=1.0) so the negative sum runs over ALL classes -> tighter cluster boundaries.
# The feature bank is frozen by default (pco_update_center_stage3=False), per Algorithm 1.

config = edict()
config.loss = "arcface"
config.margin_list = (1.0, 0.0, 0.4)   # CosFace (m3=0.4)
config.m = 0.4
config.h = 0.333
config.t_alpha = 0.01

# ----- PCO -----
config.pco_stage = 3
config.pco_proto_m1 = 0.4
config.pco_proto_m2 = 0.4
config.pco_scale = 64.0
config.pco_update_center_stage3 = False
# >>> EDIT: point this at the Stage-2 output dir.
config.pco_init_checkpoint = "/workspace/data/workspace/face_embedding/outputs/vit36_pco_stage2"

config.network = "vit_l_depth36"
config.resume = False
# >>> EDIT: a NEW output dir for this stage.
config.output = "/workspace/data/workspace/face_embedding/outputs/vit36_pco_stage3"
config.embedding_size = 512
config.sample_rate = 1.0                # NCS OFF: full negative set (Eq.12)
config.sample_rate_schedule = None
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 384
config.optimizer = "adamw"
config.lr = 0.00025
config.verbose = 2000
config.dali = False
config.gradient_acc = 8
config.save_epoch = True
config.save_all_states = True
config.use_albumentations = True

# DINO
config.pretrained_path = None
config.freeze_backbone = False
config.use_projection = False
config.dropout = 0.0

# EMA (model-weight EMA — independent of the PCO prototype EMA)
config.use_ema = False
config.ema_warmup = False
config.ema_update_after_step = 100
config.ema_decay = 0.999

# MLFLOW
config.mlflow = False
config.run_name = "vit_36_wf42m_pco_stage3"
config.experiment_name = "face_embedding"
config.frequent = 10

# DATA
config.num_rec_files = 33
config.use_synthetic_data = True
config.use_public_data = True
config.rec = "/workspace/data/workspace/face_embedding/data/dataset_71m_img_3m1_id"
config.num_classes = 3666172
config.num_image = 71654617
# Boundary Refinement is a short tail with full negatives (memory-heavy); keep it short.
config.num_epoch = 5
config.warmup_epoch = 0
config.val_targets = []
