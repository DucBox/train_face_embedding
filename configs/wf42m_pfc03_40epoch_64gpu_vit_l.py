from easydict import EasyDict as edict

config = edict()
#Loss config
config.loss = "arcface" #"adaface" or "arcface"
#ArcFace-CosFace
config.margin_list = (1.0, 0.0, 0.4) # (1.0, 0.5, 0.0) for Arcface and (1.0, 0.0, 0.4) for CosFace
#AdaFace 
config.m = 0.4
config.h = 0.333
config.t_alpha = 0.01

# config.network = "vit_l_dinov3"
config.network = "vit_l_depth36"
config.resume = False
config.output = "/workspace/data/workspace/face_embedding/outputs/vit36_webface_synthetic_public_only_new_id_71m_3m6"
config.embedding_size = 512
config.sample_rate = 0.3
# Switch PartialFC to full sampling for the last few epochs (fine-tune tail) to expose
# every negative class each step. e.g. [[0, 0.3], [65, 1.0]]. None = keep sample_rate fixed.
config.sample_rate_schedule = None
# Hard-negative-aware sampling (opt-in). See configs/base.py for the hard_neg_* knobs.
config.hard_neg_mining = False
config.hard_neg_ratio = 0.2
config.hard_neg_topk = 50
config.hard_neg_warmup_epoch = 10
config.hard_neg_refresh_interval = 2000
config.hard_neg_queue_size = 8192
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 384
config.optimizer = "adamw"
config.lr = 0.00025
config.verbose = 2000
config.dali = False
config.gradient_acc= 8
config.save_epoch = True 
config.save_all_states = True
config.use_albumentations = True
#DINO
# config.pretrained_path = "../train_results/dinov3_vitl.safetensors"
config.pretrained_path = None
config.freeze_backbone = False
config.use_projection = False
config.dropout = 0.0

#EMA Config
config.use_ema=False
config.ema_warmup=False
config.ema_update_after_step=100
config.ema_decay=0.999

#MLFLOW
config.mlflow = False
config.run_name="vit_36_webface42m_only_new_augment"
config.experiment_name="face_embedding"
config.frequent=10

#DATA
config.num_rec_files = 33
config.use_synthetic_data = True
config.use_public_data = True
config.rec = "/workspace/data/workspace/face_embedding/data/dataset_71m_img_3m1_id"
config.num_classes = 3666172 #Webface42M: 2058464
config.num_image = 71654617 #+6Msynthetic: 48827227 - WebFace42M : 42473115
config.num_epoch = 70
config.warmup_epoch = 4
config.val_targets = []
