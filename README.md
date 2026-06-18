# Training Face Embedding Model
**This source code allow you to train "Distributed Training" face embedding model on NVIDIA GPU**

## Source code
```
cd arcface_torch
```

# Training
**In order to use this script efficiently, please prepare your own dataset**

## Dataset Preparation 
**We allow you to use 2 types of Data:**

- **ImageFolder**: A directory structure where each subfolder is named after an identity (class) and contains the images of that identity. This format is commonly used with many deep learning frameworks.

**Example directory structure:**
```
  dataset/
    ├── person1/
    │ ├── img1.jpg
    │ ├── img2.jpg
    ├── person2/
    │ ├── img1.jpg
    │ ├── img2.jpg

```
- **MXNetRecord**: A record file format (.rec) commonly used with MXNet for efficient reading and storage of large datasets. It usually comes with an index file (.idx) for fast access.

**Example files:**
- train.rec
- train.idx

# Parameters Configuration
## Configuration

### Changing config at: `configs/`

**Key Training Parameters:**

- **Loss Function**: `config.loss = "arcface"` (supports "adaface" or "arcface")
- **Batch Size**: `config.batch_size = 384`
- **Gradient Accumulation**: `config.gradient_acc = 8` (effective batch size: 384 × 8 = 3,072)
- **Learning Rate**: `config.lr = 0.00025`
- **Number of Epochs**: `config.num_epoch = 60`
- **Warmup Epochs**: `config.warmup_epoch = 4`

**Loss-Specific Settings:**

*ArcFace/CosFace:*
```python
config.margin_list = (1.0, 0.0, 0.4) # (1.0, 0.5, 0.0) for ArcFace, (1.0, 0.0, 0.4) for CosFace
```

*AdaFace:*
```python
config.m = 0.4
config.h = 0.333
config.t_alpha = 0.01
```

**Model Configuration:**

- **Network**: `config.network = "vit_l_depth36"` (also supports "vit_l_dinov3")
- **Embedding Size**: `config.embedding_size = 512`
- **DINOv3 Pretrained**: `config.pretrained_path = None` (set path to use pretrained weights)
- **Freeze Backbone**: `config.freeze_backbone = False`

**Advanced Features:**

- **EMA (Exponential Moving Average)**:
```python
config.use_ema = False
config.ema_decay = 0.999
config.ema_update_after_step = 100
```

- **Data Augmentation**: `config.use_albumentations = True` (enables Albumentations library)

**Additional Settings:**
```python
config.optimizer = "adamw"
config.weight_decay = 0.1
config.fp16 = True
config.sample_rate = 0.3
config.dropout = 0.0
```

# Training

### Single GPU
```bash
python3 train_v2.py configs/wf42m_pfc03_40epoch_64gpu_vit_l
```

### Single Node Multi-GPU
```bash
torchrun --nproc_per_node=2 train_v2.py configs/wf42m_pfc03_40epoch_64gpu_vit_l
```

### Multi-Node Multi-GPU

**Node 0:**
```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=12581 train_v2.py configs/wf42m_pfc02_16gpus_r100
```

**Node 1:**
```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=12581 train_v2.py configs/wf42m_pfc02_16gpus_r100
```

**Parameters:**
- `--nproc_per_node`: Number of GPUs per node
- `--nnodes`: Total number of nodes
- `--node_rank`: Rank of the current node (0-indexed)
- `--master_addr`: IP address of the master node (Node 0)
- `--master_port`: Communication port (must be the same across all nodes)

## Contact
For any inquiries, reach out via:
- Email: [ducnq7@viettel.com.vn](mailto:ducnq7@viettel.com.vn)
- Rocket.Chat: ducnq7
- Phone: +84 912 503 111

