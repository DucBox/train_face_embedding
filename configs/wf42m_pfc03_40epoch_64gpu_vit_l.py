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
config.init_checkpoint = None  # set to "/path/to/model.pt" for backbone warm-start (fresh FC + optimizer)
config.output = "/workspace/data/workspace/face_embedding/outputs/vit36_webface_synthetic_public_only_new_id_71m_3m6"
config.embedding_size = 512
config.sample_rate = 0.3
# Switch PartialFC to full sampling for the last few epochs (fine-tune tail) to expose
# every negative class each step. e.g. [[0, 0.3], [65, 1.0]]. None = keep sample_rate fixed.
config.sample_rate_schedule = None
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
config.num_classes = 3750243  # rec_out after clean pipeline (update to 3750242 if 980k-img outlier id is dropped)
config.num_image = 95098766   # rec_out: 48.1M synthetic + 4.2M public + 42.7M crawl
config.num_epoch = 70
config.warmup_epoch = 4
config.val_targets = []
