import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable
import torch.nn.functional as F
class DINOv3Wrapper(nn.Module):
    """
    Wrapper for DINOv3 ViT-L/16 adapted for ArcFace training
    Input: [B, 3, 224, 224]
    Output: [B, num_features] (default 512)
    """
    def __init__(self, 
                 num_features=512, 
                 pretrained_path=None, 
                 freeze_backbone=False,
                 use_projection=False):
        super().__init__()
        
        # Import DINOv3 components
        import sys
        from pathlib import Path
        # Add dinov3 to path if needed
        # Assume dinov3 folder is at same level as backbones
        # Adjust this path based on your project structure
        
        from dinov3.models import vision_transformer as vits
        
        # Initialize DINOv3 ViT-L/16 with original config
        self.backbone = vits.vit_large(
            patch_size=8,
            img_size=112,  
            qkv_bias=True,
            layerscale_init=1e-5,
            norm_layer="layernorm",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            pos_embed_rope_base=100.0,
            pos_embed_rope_normalize_coords="separate",
            n_storage_tokens=4
        )
        
        self.num_features = num_features if use_projection else 1024
        self.embed_dim = 1024  
        self.use_projection = use_projection
        
        # Projection head: 1024 (DINOv3 cls token) -> num_features (ArcFace embedding)
        if use_projection:
            self.feature_proj = nn.Sequential(
                nn.Linear(self.embed_dim, num_features, bias=False),
                nn.BatchNorm1d(num_features, eps=2e-5)
            )
            # Initialize projection head
            self._init_projection()
        
        # Load pretrained weights
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
        else:
            self.backbone.init_weights()
        
        # Freeze backbone
        if freeze_backbone:
            print("Freeze backbone")
            self.freeze_backbone_weights()
    
    def _init_projection(self):
        """Initialize projection head weights"""
        for m in self.feature_proj.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def load_pretrained(self, pretrained_path):
        """
        Load pretrained weights from .pth or .safetensors
        """
        import torch
        from pathlib import Path
        
        path = Path(pretrained_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
        
        print(f"Loading pretrained weights from: {pretrained_path}")
        
        if path.suffix == '.safetensors':
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(path))
                print("Loaded .safetensors format")
            except ImportError:
                raise ImportError("safetensors not installed. Run: pip install safetensors")
        elif path.suffix in ['.pth', '.pt']:
            checkpoint = torch.load(str(path), map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'teacher' in checkpoint:
                state_dict = checkpoint['teacher']
            else:
                state_dict = checkpoint
            print("Loaded .pth format")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        state_dict = self._clean_state_dict(state_dict)
        
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded pretrained weights successfully!")
        if missing_keys:
            print(f"Missing keys (expected for projection head): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    def _clean_state_dict(self, state_dict):
        """Convert DINOv3 official checkpoint format to model format"""
        
        cleaned = {}
        qkv_temp = {}
        target_patch_size = self.backbone.patch_size
        
        for key, value in state_dict.items():
            # 1. Embeddings
            if key == 'embeddings.cls_token':
                # Shape: [dim] -> [1, 1, dim]
                cleaned['cls_token'] = value.reshape(1, 1, -1)
                continue
            elif key == 'embeddings.mask_token':
                # Shape: [dim] -> [1, dim]
                cleaned['mask_token'] = value.reshape(1, -1)
                continue
            elif key == 'embeddings.register_tokens':
                # Shape: [n_tokens, dim] -> [1, n_tokens, dim]
                v = value.view(1, -1, 1024)
                cleaned['storage_tokens'] = v
                continue
            elif key == 'embeddings.patch_embeddings.weight':
                
                current_size = value.shape[-2:]

                if current_size != (target_patch_size, target_patch_size):
                    print(f"Resize patch embed prj weights from {current_size} to {target_patch_size} ")

                    value = F.interpolate(
                        value,
                        size=(target_patch_size, target_patch_size),
                        mode='bilinear',
                        align_corners=False
                    )
                cleaned['patch_embed.proj.weight'] = value
                continue
            elif key == 'embeddings.patch_embeddings.bias':
                cleaned['patch_embed.proj.bias'] = value
                continue
            
            # 2. Layer weights
            if not key.startswith('layer.'):
                continue
                
            parts = key.split('.')
            block_idx = parts[1]
            
            # QKV weights - store temporarily for concatenation
            if '.attention.q_proj.weight' in key:
                qkv_temp[f'b{block_idx}_qw'] = value
            elif '.attention.k_proj.weight' in key:
                qkv_temp[f'b{block_idx}_kw'] = value
            elif '.attention.v_proj.weight' in key:
                qkv_temp[f'b{block_idx}_vw'] = value
            elif '.attention.q_proj.bias' in key:
                qkv_temp[f'b{block_idx}_qb'] = value
            elif '.attention.k_proj.bias' in key:
                qkv_temp[f'b{block_idx}_kb'] = value
            elif '.attention.v_proj.bias' in key:
                qkv_temp[f'b{block_idx}_vb'] = value
            
            # Output projection
            elif '.attention.o_proj.weight' in key:
                cleaned[f'blocks.{block_idx}.attn.proj.weight'] = value
            elif '.attention.o_proj.bias' in key:
                cleaned[f'blocks.{block_idx}.attn.proj.bias'] = value
            
            # MLP
            elif '.mlp.up_proj.weight' in key:
                cleaned[f'blocks.{block_idx}.mlp.fc1.weight'] = value
            elif '.mlp.up_proj.bias' in key:
                cleaned[f'blocks.{block_idx}.mlp.fc1.bias'] = value
            elif '.mlp.down_proj.weight' in key:
                cleaned[f'blocks.{block_idx}.mlp.fc2.weight'] = value
            elif '.mlp.down_proj.bias' in key:
                cleaned[f'blocks.{block_idx}.mlp.fc2.bias'] = value
            
            # LayerScale
            elif '.layer_scale1.lambda1' in key:
                cleaned[f'blocks.{block_idx}.ls1.gamma'] = value
            elif '.layer_scale2.lambda1' in key:
                cleaned[f'blocks.{block_idx}.ls2.gamma'] = value
            
            # Norms
            elif '.norm1.weight' in key:
                cleaned[f'blocks.{block_idx}.norm1.weight'] = value
            elif '.norm1.bias' in key:
                cleaned[f'blocks.{block_idx}.norm1.bias'] = value
            elif '.norm2.weight' in key:
                cleaned[f'blocks.{block_idx}.norm2.weight'] = value
            elif '.norm2.bias' in key:
                cleaned[f'blocks.{block_idx}.norm2.bias'] = value
        
        # 3. Concatenate QKV weights
        for block_idx in range(24):  # 24 layers for ViT-L
            # Weights: [dim, dim] x 3 -> [3*dim, dim]
            q_w = qkv_temp.get(f'b{block_idx}_qw')
            k_w = qkv_temp.get(f'b{block_idx}_kw')
            v_w = qkv_temp.get(f'b{block_idx}_vw')
            
            if q_w is not None and k_w is not None and v_w is not None:
                qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)
                cleaned[f'blocks.{block_idx}.attn.qkv.weight'] = qkv_weight
            
            # Bias: [dim] x 3 -> [3*dim]
            q_b = qkv_temp.get(f'b{block_idx}_qb')
            k_b = qkv_temp.get(f'b{block_idx}_kb')
            v_b = qkv_temp.get(f'b{block_idx}_vb')
            
            if q_b is not None and k_b is not None and v_b is not None:
                qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)
                cleaned[f'blocks.{block_idx}.attn.qkv.bias'] = qkv_bias
        
        print(f"Converted {len(cleaned)} keys from checkpoint")
        return cleaned
    
    def freeze_backbone_weights(self):
        print("Freezing backbone weights...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.feature_proj.parameters():
            param.requires_grad = True
        
        print(f"Backbone frozen. Trainable params: {sum(p.numel() for p in self.feature_proj.parameters()):,}")
    
    def unfreeze_backbone(self):
        print("Unfreezing backbone weights...")
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):

        features_dict = self.backbone.forward_features(x)
        
        cls_token = features_dict["x_norm_clstoken"]
        print(f"Cls tokn shape: {cls_token.shape}")
        if self.use_projection:
            print(f"Using projection")
            embedding = self.feature_proj(cls_token)
            print(f"After projection: {embedding.shape}")
        else:
            print("No projection")
            embedding = cls_token
        print(f"Embedding shape: {embedding.shape}")
        return embedding
    
    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters())