import json
from multiprocessing import context
import re
import os.path
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, get_t_distribution, slice_t_distribution, sample_t, load_state_dict
from utils.offloading import ModelOffloader
from .t5 import T5EncoderModel
from .vae2_1 import Wan2_1_VAE
from .vae2_2 import Wan2_2_VAE
from .model import (
    WanModel, sinusoidal_embedding_1d
)
from .clip import CLIPModel
from . import configs as wan_configs

from models.base import MLP_Clip2Added_kv
from transformers import AutoImageProcessor
from models.base import MLP_Clip2Added_kv, MLP_ControlLatents2Added_kv

KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head', 'modulation']


class WanModelFromSafetensors(WanModel):
    @classmethod
    def from_pretrained(
        cls,
        weights_file,
        config,
        torch_dtype=torch.bfloat16,
        transformer_dtype=torch.bfloat16,
    ):
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        with init_empty_weights():
            model = cls(**config)

        state_dict = load_state_dict(weights_file)
        state_dict = {
            re.sub(r'^model\.diffusion_model\.', '', k): v for k, v in state_dict.items()
        }

        for name, param in model.named_parameters():
            dtype_to_use = torch_dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
            set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        return model


def vae_encode(tensor, vae):
    return vae.model.encode(tensor, vae.scale)


# Wrapper to hold both VAE and CLIP, so we can move both to/from GPU together.
# Wrapper to hold VAE, CLIP, and SigLIP
class VaeClipAndSiglip(nn.Module):
    def __init__(self, vae, clip, siglip=None, siglip_model_path="/home/zhengtianyu/yaowangzi/siglip-so400m-patch14-384/"):
        super().__init__()
        self.vae = vae
        self.clip = clip
        self.siglip = siglip
        self.siglip_processor = AutoImageProcessor.from_pretrained(siglip_model_path)


class WanPipeline(BasePipeline):
    name = 'wan'
    framerate = 16
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['WanAttentionBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.cache_text_embeddings = self.model_config.get('cache_text_embeddings', True)
        self.t_dist = get_t_distribution(self.model_config).to('cuda')
        if config['adapter']['type']=='ip-adapter':
            self.ip_adapter=True
        else:
            self.ip_adapter=False

        # The official Wan top-level checkpoint folder. Must exist.
        ckpt_dir = Path(self.model_config['ckpt_path'])
        dtype = self.model_config['dtype']

        # transformer_path will either be a directory containing safetensors files, or directly point to a safetensors file
        self.transformer_path = Path(self.model_config.get('transformer_path', ckpt_dir))
        if self.transformer_path.is_dir():
            # If it's a directory, we assume the config JSON exists inside it, along with the safetensors files.
            self.original_model_config_path = self.transformer_path / 'config.json'
            safetensors_files = list(self.transformer_path.glob('*.safetensors'))
        else:
            # If it's a single file, the config JSON is assumed to be in the top-level checkpoint folder.
            self.original_model_config_path = ckpt_dir / 'config.json'
            if not self.original_model_config_path.exists():
                # Wan2.2 has subdirectories for the model. Automatically handle that.
                self.original_model_config_path = ckpt_dir / 'low_noise_model' / 'config.json'
            safetensors_files = [self.transformer_path]

        # get all weight keys
        weight_keys = set()
        for shard in safetensors_files:
            with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weight_keys.add(re.sub(r'^model\.diffusion_model\.', '', k))

        # SkyReels V2 uses 24 FPS. There seems to be no better way to autodetect this.
        if 'skyreels' in ckpt_dir.name.lower():
            skyreels = True
            self.framerate = 24
        else:
            skyreels = False

        with open(self.original_model_config_path) as f:
            self.json_config = json.load(f)

        model_type = self.json_config['model_type']
        model_dim = self.json_config['dim']

        def autodetect_error():
            raise RuntimeError(f'Could not autodetect model variant. model_type={model_type}, model_dim={model_dim}')

        if model_type == 't2v':
            if skyreels:
                # FPS is different so make sure to use a new cache dir
                self.name = 'skyreels_v2'
            if model_dim == 1536:
                wan_config = wan_configs.t2v_1_3B
            elif model_dim == 5120:
                # This config also works with Wan2.2 T2V.
                wan_config = wan_configs.t2v_14B
            else:
                autodetect_error()
        elif model_type == 'i2v':
            if 'blocks.0.cross_attn.k_img.weight' not in weight_keys:
                # Wan2.2 I2V
                model_type = 'i2v_v2'
                self.name = 'wan2.2_i2v'
                if model_dim == 5120:
                    wan_config = wan_configs.i2v_A14B
                else:
                    autodetect_error()
            else:
                if skyreels:
                    self.name = 'skyreels_v2_i2v'
                else:
                    self.name = 'wan_i2v'
                if model_dim == 1536: # There is no official i2v 1.3b model, but there is https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP
                    # This is a hack,
                    wan_config = wan_configs.t2v_1_3B
                    # The following lines are taken from https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_i2v_14B.py
                    wan_config.clip_model = 'clip_xlm_roberta_vit_h_14'
                    wan_config.clip_dtype = torch.float16
                    wan_config.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
                elif model_dim == 5120:
                    wan_config = wan_configs.i2v_14B
                else:
                    autodetect_error()
        elif model_type == 'flf2v':
            assert not skyreels
            self.name = 'wan_flf2v'
            if model_dim == 5120:
                wan_config = wan_configs.i2v_14B  # flf2v is same config as i2v
            else:
                autodetect_error()
        elif model_type == 'ti2v':
            self.name = 'wan2.2_5b'
            self.framerate = 24
            if model_dim == 3072:
                wan_config = wan_configs.ti2v_5B
            else:
                autodetect_error()
        else:
            raise RuntimeError(f'Unknown model_type: {model_type}')

        self.model_type = model_type
        self.json_config['model_type'] = model_type  # to handle the special i2v_v2 type we introduced

        # This is the outermost class, which isn't a nn.Module
        t5_model_path = self.model_config['llm_path'] if self.model_config.get('llm_path', None) else os.path.join(ckpt_dir, wan_config.t5_checkpoint)
        self.text_encoder = T5EncoderModel(
            text_len=wan_config.text_len,
            dtype=dtype,
            device='cpu',
            checkpoint_path=t5_model_path,
            tokenizer_path=ckpt_dir / wan_config.t5_tokenizer,
            shard_fn=None,
        )
        if self.model_config.get('text_encoder_fp8', False):
            for name, p in self.text_encoder.model.named_parameters():
                if p.ndim == 2 and not ('token_embedding' in name or 'pos_embedding' in name):
                    p.data = p.data.to(torch.float8_e4m3fn)
        self.text_encoder.model.requires_grad_(False)

        vae_class = Wan2_2_VAE if model_type == 'ti2v' else Wan2_1_VAE
        # Same here, this isn't a nn.Module.
        self.vae = vae_class(
            vae_pth=ckpt_dir / wan_config.vae_checkpoint,
            device='cpu',
            dtype=dtype,
        )
        self.vae.model.to(dtype)
        # These tensors need to be on the device the VAE will be moved to during caching.
        self.vae.scale = [entry.to('cuda') for entry in self.vae.scale]

        if model_type in ('i2v', 'flf2v'):
            self.clip = CLIPModel(
                dtype=dtype,
                device='cpu',
                checkpoint_path=ckpt_dir / wan_config.clip_checkpoint,
                tokenizer_path=ckpt_dir / wan_config.clip_tokenizer,
            )

        # ==================== 新增：加载 SigLIP ====================
        # 如果是 ti2v 模型，加载 SigLIP 用于风格图编码
        if model_type == 'ti2v':
            from transformers import SiglipVisionModel, AutoImageProcessor
            
            siglip_path = self.model_config.get(
                'siglip_path', 
                '/home/zhengtianyu/yaowangzi/siglip-so400m-patch14-384/'
            )
            
            print(f"Loading SigLIP from {siglip_path}")
            self.siglip_encoder = SiglipVisionModel.from_pretrained(siglip_path)
            self.siglip_processor = AutoImageProcessor.from_pretrained(siglip_path)
            self.siglip_encoder.requires_grad_(False).eval()
            # 初始化时在CPU，caching时会移到CUDA
        else:
            self.siglip_encoder = None
            self.siglip_processor = None
        
        # 在 siglip_encoder 初始化之后
        if model_type == 'ti2v':
            # 初始化 Control MLP
            self.control_mlp = MLP_ControlLatents2Added_kv(
                latent_channels=16,  # VAE latent channels
                dim=self.json_config['dim'],  # 3072 for ti2v_5B
            )
            print(f"Initialized Control MLP for control_latents injection")
    # delay loading transformer to save RAM
    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if self.transformer_path.is_file():
            self.transformer = WanModelFromSafetensors.from_pretrained(
                self.transformer_path,
                self.json_config,
                torch_dtype=dtype,
                transformer_dtype=transformer_dtype,
            )
        else:
            with init_empty_weights():
                self.transformer = WanModel.from_config(self.json_config)
            state_dict = {}
            for shard in self.transformer_path.glob('*.safetensors'):
                with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            for name, param in self.transformer.named_parameters():
                dtype_to_use = dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                set_module_tensor_to_device(self.transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        self.transformer.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        vae = self.vae.model
        clip = self.clip.model if self.model_type in ('i2v', 'flf2v') else None
        siglip = self.siglip_encoder if self.model_type == 'ti2v' else None
        return VaeClipAndSiglip(vae, clip, siglip)

    def get_text_encoders(self):
        # Return the inner nn.Module
        if self.cache_text_embeddings:
            return [self.text_encoder.model]
        else:
            return []

    def save_adapter(self, save_dir, peft_state_dict=None):
        if self.ip_adapter:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.MLP_Clip2Added_kv.state_dict(), save_dir / "mlp_clip2added_kv.pt")
            torch.save(self.control_mlp.state_dict(), save_dir / "control_mlp.pt")
            return
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def load_adapter_weights(self, adapter_path):
        path = Path(adapter_path) / "mlp_clip2added_kv.pt"

        dim = 1536                 # 例子：cross_attention_dim
        id_embeddings_dim = 1152   # 例子：SigLIP 输出维度
        num_tokens = 16            # 例子：你当时设置的 token 数

        self.MLP_Clip2Added_kv = MLP_Clip2Added_kv(dim=dim, id_embeddings_dim=id_embeddings_dim, num_tokens=num_tokens)

        state = torch.load(path, map_location="cpu")
        load_info = self.MLP_Clip2Added_kv.load_state_dict(state, strict=True)   # 不想严格就 strict=False
        print(getattr(load_info, "missing_keys", []), getattr(load_info, "unexpected_keys", []))
        control_mlp_path = Path(adapter_path) / "control_mlp.pt"
        if not hasattr(self, 'control_mlp'):
            self.control_mlp = MLP_ControlLatents2Added_kv(
                latent_channels=16,
                dim=self.json_config['dim']
            )
        state = torch.load(control_mlp_path, map_location="cpu")
        self.control_mlp.load_state_dict(state, strict=True)
        print(f"Loaded control MLP from {control_mlp_path}")
    
    def save_model(self, save_dir, state_dict):
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        if self.model_type == 'ti2v':
            round_side = 32
        else:
            round_side = 16
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
            round_height=round_side,
            round_width=round_side,
        )

    def get_call_vae_fn(self, vae_clip_siglip):
        
        def fn(tensor, control_tensor=None, condition_images=None):
            vae = vae_clip_siglip.vae
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            
            # 编码label视频
            latents = vae_encode(tensor, self.vae)
            ret = {'latents': latents}
            
            # 编码control视频 (destyle视频)
            if control_tensor is not None:
                control_tensor = control_tensor.to(p.device, p.dtype)
                control_latents = vae_encode(control_tensor, self.vae)
                ret['control_latents'] = control_latents
            
            # 编码条件图片（用 SigLIP）
            if condition_images is not None and vae_clip_siglip.siglip is not None:
                siglip = vae_clip_siglip.siglip
                siglip_processor = vae_clip_siglip.siglip_processor  # ← 需要添加这个
                
                # ← 修改：处理 List[List[PIL.Image]] 输入
                all_embeds = []
                for pil_images in condition_images:  # 遍历batch中的每个样本
                    if pil_images is None or any(img is None for img in pil_images):
                        # 失败情况：用零向量
                        embed_dim = siglip.config.hidden_size
                        all_embeds.append(torch.zeros(3, embed_dim, device=p.device))
                        continue
                    
                    # ← 使用 image_processor，与参考代码一致
                    inputs = siglip_processor(images=pil_images, return_tensors="pt")
                    pixel_values_siglip = inputs["pixel_values"].to(p.device, non_blocking=True)
                    
                    with torch.no_grad():
                        siglip_outputs = siglip(pixel_values=pixel_values_siglip)
                        
                        # ← 与参考代码一致：优先使用 pooler_output
                        if hasattr(siglip_outputs, "pooler_output") and siglip_outputs.pooler_output is not None:
                            image_embeds = siglip_outputs.pooler_output  # (3, D)
                        else:
                            image_embeds = siglip_outputs.last_hidden_state.mean(dim=1)  # (3, D)
                    
                    all_embeds.append(image_embeds)
                
                # Stack成 (B, 3, D)
                image_embeds = torch.stack(all_embeds)
                ret['condition_image_embeds'] = image_embeds.to(torch.float32)
            
            return ret
        
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # Args are lists
            p = next(text_encoder.parameters())
            ids, mask = self.text_encoder.tokenizer(caption, return_mask=True, add_special_tokens=True)
            ids = ids.to(p.device)
            mask = mask.to(p.device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            with torch.autocast(device_type=p.device.type, dtype=p.dtype):
                text_embeddings = text_encoder(ids, mask)
                return {'text_embeddings': text_embeddings, 'seq_lens': seq_lens}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        #==================== 数据验证代码 ====================
        # print("\n" + "="*80)
        # print("📊 DATA VALIDATION - prepare_inputs")
        # print("="*80)
        
        # # 1. 检查 inputs 包含的键
        # print(f"\n🔑 Input keys: {list(inputs.keys())}")
        
        # # 2. 检查你的新数据
        # expected_keys = ['latents', 'control_latents', 'condition_image_embeds', 
        #                 'mask', 'text_embeddings', 'seq_lens', 'caption']
        
        # for key in expected_keys:
        #     if key in inputs:
        #         value = inputs[key]
        #         if isinstance(value, torch.Tensor):
        #             print(f"\n✅ {key}:")
        #             print(f"   Shape: {value.shape}")
        #             print(f"   Dtype: {value.dtype}")
        #             print(f"   Device: {value.device}")
        #             print(f"   Range: [{value.min():.4f}, {value.max():.4f}]")
        #             print(f"   Mean: {value.mean():.4f}, Std: {value.std():.4f}")
        #             # 检查是否有NaN或Inf
        #             if torch.isnan(value).any():
        #                 print(f"   ⚠️  WARNING: Contains NaN!")
        #             if torch.isinf(value).any():
        #                 print(f"   ⚠️  WARNING: Contains Inf!")
        #         elif isinstance(value, list):
        #             print(f"\n✅ {key}: List with {len(value)} items")
        #             if len(value) > 0:
        #                 print(f"   First item type: {type(value[0])}")
        #                 if isinstance(value[0], str):
        #                     print(f"   Example: '{value[0][:100]}...'")
        #         else:
        #             print(f"\n✅ {key}: {type(value)} = {value}")
        #     else:
        #         print(f"\n❌ {key}: NOT FOUND in inputs")
        
        # # 3. 特别检查你的新数据维度是否符合预期
        # if 'latents' in inputs and 'control_latents' in inputs:
        #     latents = inputs['latents']
        #     control_latents = inputs['control_latents']
        #     condition_image_embeds = inputs.get('condition_image_embeds')
            
        #     print(f"\n📐 Shape Check:")
        #     print(f"   latents:               {latents.shape}")
        #     print(f"   control_latents:       {control_latents.shape}")
        #     if condition_image_embeds is not None:
        #         print(f"   condition_image_embeds: {condition_image_embeds.shape}")
                
        #     # 验证形状一致性
        #     assert latents.shape == control_latents.shape, \
        #         f"Shape mismatch: latents {latents.shape} != control_latents {control_latents.shape}"
            
        #     if condition_image_embeds is not None:
        #         bs, num_imgs, c, t, h, w = condition_image_embeds.shape
        #         print(f"   ✓ Condition images: {bs} batches × {num_imgs} images × ({c}, {t}, {h}, {w})")
        #         assert num_imgs == 3, f"Expected 3 condition images, got {num_imgs}"
        #         assert t == 1, f"Expected single-frame images, got {t} frames"
        
        # print("\n" + "="*80)
        # print("✅ Validation complete - proceeding with training")
        # print("="*80 + "\n")
        
        # ==================== 原始代码继续 ====================
        latents = inputs['latents'].float()
        mask = None
        # ... 后续代码保持不变 ...
        latents = inputs['latents'].float()
        mask = None
        y = None
        # No CLIP for i2v_v2 (Wan2.2)
        clip_context = None

        if self.cache_text_embeddings:
            text_embeddings_or_ids = inputs['text_embeddings']
            seq_lens_or_text_mask = inputs['seq_lens']
        else:
            text_embeddings_or_ids, seq_lens_or_text_mask = self.text_encoder.tokenizer(inputs['caption'], return_mask=True, add_special_tokens=True)

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

        t = self.t_dist

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        t = slice_t_distribution(t, min_t=self.model_config.get('min_t', 0.0), max_t=self.model_config.get('max_t', 1.0))
        t = sample_t(t, bs, quantile=timestep_quantile).to(latents.device)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        # timestep input to model needs to be in range [0, 1000]
        t = t * 1000

        # 随机从 3 张图中选 2 张
        condition_image_embeds = inputs['condition_image_embeds']  # (B, 3, 1152)
        if condition_image_embeds is not None and condition_image_embeds.shape[1] == 3:
            # 随机选择 2 个索引
            indices = torch.randperm(3)[:2]  # 从 [0,1,2] 中随机选 2 个
            condition_image_embeds = condition_image_embeds[:, indices, :]  # (B, 2, 1152)
        
        return (
            (x_t, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_context, 
            condition_image_embeds, inputs.get('control_latents')),
            (target, mask),
        )

    # def to_layers(self):
    #     transformer = self.transformer
    #     text_encoder = None if self.cache_text_embeddings else self.text_encoder.model
    #     layers = [InitialLayer(transformer, text_encoder)]
    #     for i, block in enumerate(transformer.blocks):
    #         layers.append(TransformerLayer(block, i, self.offloader))
    #     layers.append(FinalLayer(transformer))
    #     return layers

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder.model

        layers = []

        # IP-Adapter MLP layer
        if hasattr(self, 'MLP_Clip2Added_kv') and isinstance(self.MLP_Clip2Added_kv, nn.Module):
            layers.append(IPAdapterMLPLayer(self.MLP_Clip2Added_kv))
        
        # Control MLP layer (新增)
        if hasattr(self, 'control_mlp') and isinstance(self.control_mlp, nn.Module):
            layers.append(ControlLatentsMLPLayer(self.control_mlp))

        layers.append(InitialLayer(transformer, text_encoder))
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers


    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()

class ControlLatentsMLPLayer(nn.Module):
    """
    处理 control_latents，将其转换为 added_kv_control
    输入 9 项: (..., added_kv, added_kv_lens, control_latents)
    输出 11 项: (..., added_kv, added_kv_lens, added_kv_control, added_kv_control_lens)
    """
    def __init__(self, control_mlp: nn.Module):
        super().__init__()
        self.mlp = control_mlp

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        from models.base import make_contiguous
        
        L = len(inputs)
        if L == 9:
            # 拆解：前 8 项 + control_latents
            *first_8, control_latents = inputs
            
            if control_latents is not None:
                # 通过 MLP 处理
                added_kv_control = self.mlp(control_latents)  # (B, T*H*W, dim)
                added_kv_control_lens = torch.full(
                    (added_kv_control.size(0),), added_kv_control.size(1),
                    device=added_kv_control.device, dtype=torch.long
                )
            else:
                added_kv_control = None
                added_kv_control_lens = None
            
            # 输出 11 项
            return make_contiguous(*first_8, added_kv_control, added_kv_control_lens)
        
        elif L in (6, 7, 8, 10, 11):
            # 其他情况直接透传
            return inputs
        else:
            raise ValueError(f"ControlLatentsMLPLayer: unexpected inputs length={L}")

# wan.py
class IPAdapterMLPLayer(nn.Module):
    """
    前置层：在 InitialLayer 之前，把 id_embeds -> (added_kv, added_kv_lens)
    约定：
      - 老路径：6 项 (x, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_fea) -> 直接透传
      - IPA 路径：7 项 (..., clip_fea, id_embeds) -> 转成 8 项 (..., clip_fea, added_kv, added_kv_lens)
      - 若已是 8/9 项（可能上游已处理），则直接透传
    """
    def __init__(self, mlp_clip2added_kv: nn.Module):
        super().__init__()
        self.mlp = mlp_clip2added_kv  # WanPipeline.MLP_Clip2Added_kv

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        L = len(inputs)
        if L == 7:
            # 拆出第7项的 id_embeds
            x, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_fea, id_embeds = inputs
            added_kv = self.mlp(id_embeds)  # (B, T, C)
            added_kv_lens = torch.full(
                (added_kv.size(0),), added_kv.size(1), device=added_kv.device, dtype=torch.long
            )
            from models.base import make_contiguous
            return make_contiguous(
                x, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_fea, added_kv, added_kv_lens
            )
        elif L in (6, 8, 9):
            # 6: 老路径；8/9：已带 added_kv 的路径 -> 原样透传
            return inputs
        else:
            raise ValueError(f"IPAdapterMLPLayer: unexpected inputs length={L}")


# ==== wan.py ====
class InitialLayer(nn.Module):
    def __init__(self, model, text_encoder):
        super().__init__()
        self.patch_embedding = model.patch_embedding
        self.time_embedding = model.time_embedding
        self.text_embedding = model.text_embedding
        self.time_projection = model.time_projection
        self.i2v = (model.model_type == 'i2v')
        self.i2v_v2 = (model.model_type == 'i2v_v2')
        self.flf2v = (model.model_type == 'flf2v')
        if self.i2v or self.flf2v:
            self.img_emb = model.img_emb
        self.text_encoder = text_encoder
        self.freqs = model.freqs
        self.freq_dim = model.freq_dim
        self.dim = model.dim
        self.text_len = model.text_len

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # --- 向后兼容的输入解析 ---
        # 旧版: (x, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_fea)
        # 新版: ( ... , clip_fea, added_kv, added_kv_lens )
        x, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_fea, added_kv, added_kv_lens, added_kv_control, added_kv_control_lens = inputs

        for item in (x, y, t):
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        bs, channels, f, h, w = x.shape
        if clip_fea.numel() == 0:
            clip_fea = None

        # --- 文本上下文（与原版一致） ---
        if self.text_encoder is not None:
            assert not torch.is_floating_point(text_embeddings_or_ids)
            with torch.no_grad():
                context = self.text_encoder(text_embeddings_or_ids, seq_lens_or_text_mask)
            context.requires_grad_(True)
            text_seq_lens = seq_lens_or_text_mask.gt(0).sum(dim=1).long()
        else:
            context = text_embeddings_or_ids
            text_seq_lens = seq_lens_or_text_mask
        context = [emb[:length] for emb, length in zip(context, text_seq_lens)]

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # --- patchify 与 pad（与原版一致） ---
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens.max()
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # --- time embeddings（与原版一致） ---
        time_embed_seq_len = seq_len
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            time_embed_seq_len = 1  # broadcast
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, time_embed_seq_len)).to(x.device, torch.float32)
        )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # --- 文本 embedding 映射与（可选）CLIP 拼接（与原版一致） ---
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )
        if self.i2v or self.flf2v:
            assert clip_fea is not None
            if self.flf2v:
                self.img_emb.emb_pos.data = self.img_emb.emb_pos.data.to(clip_fea.device, torch.float32)
                clip_fea = clip_fea.view(-1, 257, 1280)
            context_clip = self.img_emb(clip_fea)  # bs x 257(x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        # --- 所有张量置于同一设备（与原版一致） ---
        seq_lens = seq_lens.to(x.device)
        grid_sizes = grid_sizes.to(x.device)

        # 有 control 参数
        added_kv_control = inputs[8] if len(inputs) > 8 else None
        added_kv_control_lens = inputs[9] if len(inputs) > 9 else None
        return make_contiguous(x, e, e0, seq_lens, grid_sizes, self.freqs, context,
                            added_kv, added_kv_lens, added_kv_control, added_kv_control_lens)
    
class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, added_kv, added_kv_lens, added_kv_control, added_kv_control_lens = inputs

        self.offloader.wait_for_block(self.block_idx)
        x = self.block(
            x, e0, seq_lens, grid_sizes, freqs, context, None,
            added_kv=added_kv, added_kv_lens=added_kv_lens,
            added_kv_control=added_kv_control, added_kv_control_lens=added_kv_control_lens  # 新增
        )
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, e, e0, seq_lens, grid_sizes, freqs, context,
                              added_kv, added_kv_lens, added_kv_control, added_kv_control_lens)

class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, _added_kv, _added_kv_lens, _added_kv_ctrl, _added_kv_ctrl_lens = inputs

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x, dim=0)
