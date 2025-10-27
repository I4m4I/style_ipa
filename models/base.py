from pathlib import Path
import re
import tarfile

import peft
import torch
from torch import nn
import torch.nn.functional as F
import safetensors.torch
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms
import imageio

from utils.common import is_main_process, VIDEO_EXTENSIONS, round_to_nearest_multiple, round_down_to_multiple


def make_contiguous(*tensors):
    return tuple(x.contiguous() for x in tensors)


def extract_clips(video, target_frames, video_clip_mode):
    # video is (channels, num_frames, height, width)
    frames = video.shape[1]
    if frames < target_frames:
        # TODO: think about how to handle this case. Maybe the video should have already been thrown out?
        print(f'video with shape {video.shape} is being skipped because it has less than the target_frames')
        return []

    if video_clip_mode == 'single_beginning':
        return [video[:, :target_frames, ...]]
    elif video_clip_mode == 'single_middle':
        start = int((frames - target_frames) / 2)
        assert frames-start >= target_frames
        return [video[:, start:start+target_frames, ...]]
    elif video_clip_mode == 'multiple_overlapping':
        # Extract multiple clips so we use the whole video for training.
        # The clips might overlap a little bit. We never cut anything off the end of the video.
        num_clips = ((frames - 1) // target_frames) + 1
        start_indices = torch.linspace(0, frames-target_frames, num_clips).int()
        return [video[:, i:i+target_frames, ...] for i in start_indices]
    else:
        raise NotImplementedError(f'video_clip_mode={video_clip_mode} is not recognized')


def convert_crop_and_resize(pil_img, width_and_height):
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')

    # add white background for transparent images
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    return ImageOps.fit(pil_img, width_and_height)


class PreprocessMediaFile:
    def __init__(self, config, support_video=False, framerate=None, round_height=16, round_width=16, round_frames=4):
        self.config = config
        self.video_clip_mode = config.get('video_clip_mode', 'single_beginning')
        print(f'using video_clip_mode={self.video_clip_mode}')
        self.pil_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.support_video = support_video
        self.framerate = framerate
        print(f'using framerate={self.framerate}')
        self.round_height = round_height
        self.round_width = round_width
        self.round_frames = round_frames
        if self.support_video:
            assert self.framerate
        self.tarfile_map = {}

    def __del__(self):
        for tar_f in self.tarfile_map.values():
            tar_f.close()

    def __call__(self, spec, mask_filepath, size_bucket=None):
        is_video = (Path(spec[1]).suffix in VIDEO_EXTENSIONS)

        if spec[0] is None:
            tar_f = None
            filepath_or_file = str(spec[1])
        else:
            tar_filename = spec[0]
            tar_f = self.tarfile_map.setdefault(tar_filename, tarfile.TarFile(tar_filename))
            filepath_or_file = tar_f.extractfile(str(spec[1]))

        if is_video:
            assert self.support_video
            num_frames = 0
            for frame in imageio.v3.imiter(filepath_or_file, fps=self.framerate):
                num_frames += 1
                height, width = frame.shape[:2]
            video = imageio.v3.imiter(filepath_or_file, fps=self.framerate)
        else:
            num_frames = 1
            pil_img = Image.open(filepath_or_file)
            height, width = pil_img.height, pil_img.width
            video = [pil_img]

        if size_bucket is not None:
            size_bucket_width, size_bucket_height, size_bucket_frames = size_bucket
        else:
            size_bucket_width, size_bucket_height, size_bucket_frames = width, height, num_frames

        height_rounded = round_to_nearest_multiple(size_bucket_height, self.round_height)
        width_rounded = round_to_nearest_multiple(size_bucket_width, self.round_width)
        frames_rounded = round_down_to_multiple(size_bucket_frames - 1, self.round_frames) + 1
        resize_wh = (width_rounded, height_rounded)

        if mask_filepath:
            mask_img = Image.open(mask_filepath).convert('RGB')
            img_hw = (height, width)
            mask_hw = (mask_img.height, mask_img.width)
            if mask_hw != img_hw:
                raise ValueError(
                    f'Mask shape {mask_hw} was not the same as image shape {img_hw}.\n'
                    f'Image path: {spec[1]}\n'
                    f'Mask path: {mask_filepath}'
                )
            mask_img = ImageOps.fit(mask_img, resize_wh)
            mask = torchvision.transforms.functional.to_tensor(mask_img)[0].to(torch.float16)  # use first channel
        else:
            mask = None

        resized_video = torch.empty((num_frames, 3, height_rounded, width_rounded))
        for i, frame in enumerate(video):
            if not isinstance(frame, Image.Image):
                frame = torchvision.transforms.functional.to_pil_image(frame)
            cropped_image = convert_crop_and_resize(frame, resize_wh)
            resized_video[i, ...] = self.pil_to_tensor(cropped_image)

        if hasattr(filepath_or_file, 'close'):
            filepath_or_file.close()

        if not self.support_video:
            return [(resized_video.squeeze(0), mask)]

        # (num_frames, channels, height, width) -> (channels, num_frames, height, width)
        resized_video = torch.permute(resized_video, (1, 0, 2, 3))
        if not is_video:
            return [(resized_video, mask)]
        else:
            videos = extract_clips(resized_video, frames_rounded, self.video_clip_mode)
            return [(video, mask) for video in videos]

import math

# class MLP_Clip2Added_kv(nn.Module):
#     """
#     将图像向量(例如 1152 维 SigLIP)投影成 num_tokens 个、每个维度为 cross_attention_dim 的 tokens。
#     """
#     def __init__(self, dim: int, id_embeddings_dim: int, num_tokens: int, siglip_pooling: bool=True):
#         super().__init__()
#         self.dim = dim
#         self.id_embeddings_dim = id_embeddings_dim
#         self.num_tokens = num_tokens
#         self.siglip_pooling = siglip_pooling
#         self.proj = nn.Sequential(
#             nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
#             nn.GELU(),
#             nn.Linear(id_embeddings_dim * 2, dim * num_tokens),
#         )
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, id_embeds: torch.Tensor):
#         x = self.proj(id_embeds)                                   # (B, num_tokens * C)
#         x = x.view(-1, self.num_tokens, self.dim)  # (B, T, C)
#         return self.norm(x)

class MLP_Clip2Added_kv(nn.Module):
    """
    将图像向量(例如 1152 维 SigLIP)投影成 num_tokens 个、每个维度为 cross_attention_dim 的 tokens。
    """
    def __init__(self, dim: int, id_embeddings_dim: int, num_tokens: int, siglip_pooling: bool=True):
        super().__init__()
        self.dim = dim
        self.id_embeddings_dim = id_embeddings_dim
        self.num_tokens = num_tokens
        self.siglip_pooling = siglip_pooling
        if siglip_pooling:
            self.proj = nn.Sequential(
                nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
                nn.GELU(),
                nn.Linear(id_embeddings_dim * 2, dim * num_tokens),
            )
            self.norm = nn.LayerNorm(dim)
        else:
            # ---- 固定输入长度与网格 ----
            self.L = 728
            # 728 = 26 * 28（正好，不需要 padding）
            self.H, self.W = 26, 28
            self.total = self.H * self.W  # = 728

            # ---- 预先分解 num_tokens -> (H_out, W_out) ----
            self.H_out, self.W_out = self._factorize_to_hw(self.num_tokens)

            # ---- 两层 MLP: D -> 2D -> dim ----
            self.mlp = nn.Sequential(
                nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
                nn.GELU(),
                nn.Linear(id_embeddings_dim * 2, dim),
            )
            self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _factorize_to_hw(n: int):
        """把 n 分解成尽量接近方形的 (H, W)。若为质数则返回 (1, n)。"""
        h = int(math.floor(math.sqrt(n)))
        while h > 1:
            if n % h == 0:
                return h, n // h
            h -= 1
        return 1, n

    def forward(self, id_embeds: torch.Tensor):
        if self.siglip_pooling:
            x = self.proj(id_embeds)                                   # (B, num_tokens * C)
            x = x.view(-1, self.num_tokens, self.dim)  # (B, T, C)
            return self.norm(x)
        else:
            """
            id_embeds: [B, L, D], D == id_embeddings_dim 且 L == 728
            返回: [B, num_tokens, dim]
            """
            assert id_embeds.dim() == 3, f"Expect [B, L, D], got {tuple(id_embeds.shape)}"
            B, L, D = id_embeds.shape
            assert D == self.id_embeddings_dim, f"D mismatch: got {D}, expected {self.id_embeddings_dim}"
            assert L == self.L, f"L mismatch: got {L}, expected {self.L} (26x28)."

            # [B, L, D] -> [B, H, W, D] -> [B, D, H, W]
            x = id_embeds.view(B, self.H, self.W, D).permute(0, 3, 1, 2)

            # 自适应平均池化到 (H_out, W_out) -> [B, D, H_out, W_out]
            x = F.adaptive_avg_pool2d(x, output_size=(self.H_out, self.W_out))

            # 展平空间 -> token 维: [B, num_tokens, D]
            x = x.permute(0, 2, 3, 1).reshape(B, self.H_out * self.W_out, D)

            # 两层 MLP + LN: [B, num_tokens, dim]
            x = self.mlp(x)
            x = self.norm(x)
            return x

class BasePipeline:
    framerate = None

    def load_diffusion_model(self):
        pass

    def get_vae(self):
        raise NotImplementedError()

    def get_text_encoders(self):
        raise NotImplementedError()

    # def configure_adapter(self, adapter_config):
    #     target_linear_modules = set()
    #     for name, module in self.transformer.named_modules():
    #         if module.__class__.__name__ not in self.adapter_target_modules:
    #             continue
    #         for full_submodule_name, submodule in module.named_modules(prefix=name):
    #             if isinstance(submodule, nn.Linear):
    #                 target_linear_modules.add(full_submodule_name)
    #     target_linear_modules = list(target_linear_modules)

    #     adapter_type = adapter_config['type']
    #     if adapter_type == 'lora':
    #         peft_config = peft.LoraConfig(
    #             r=adapter_config['rank'],
    #             lora_alpha=adapter_config['alpha'],
    #             lora_dropout=adapter_config['dropout'],
    #             bias='none',
    #             target_modules=target_linear_modules
    #         )
    #     else:
    #         raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
    #     self.peft_config = peft_config
    #     self.lora_model = peft.get_peft_model(self.transformer, peft_config)
    #     if is_main_process():
    #         self.lora_model.print_trainable_parameters()
    #     for name, p in self.transformer.named_parameters():
    #         p.original_name = name
    #         if p.requires_grad:
    #             p.data = p.data.to(adapter_config['dtype'])

    def configure_adapter(self, adapter_config):
        """
        支持两种模式：
        - adapter.type == 'lora'：保持现有逻辑，用 peft 给 self.transformer 包 LoRA
        - adapter.type == 'ip-adapter'：不做 LoRA 包装；冻结 base transformer 参数，后续仅训练你挂载的 IP-Adapter 模块
        """
        target_linear_modules = set()
        for name, module in self.transformer.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.add(full_submodule_name)
        target_linear_modules = list(target_linear_modules)

        adapter_type = adapter_config['type']

        if adapter_type == 'lora':
            # === 维持原有 LoRA 逻辑 ===
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
            self.peft_config = peft_config
            self.lora_model = peft.get_peft_model(self.transformer, peft_config)
            if is_main_process():
                self.lora_model.print_trainable_parameters()

            # 仅把需要训练的参数转到期望的 dtype（保持原逻辑）
            for name, p in self.transformer.named_parameters():
                p.original_name = name
                if p.requires_grad:
                    p.data = p.data.to(adapter_config['dtype'])

        elif adapter_type == 'ip-adapter':
            # === 新增：IP-Adapter 模式，不包 LoRA ===
            # 不创建 peft_config / lora_model；冻结 base transformer，后续只训练你挂的 adapter 模块
            self.peft_config = None
            self.lora_model = None
            self.MLP_Clip2Added_kv=MLP_Clip2Added_kv(dim=self.transformer.dim,id_embeddings_dim=1152,num_tokens=adapter_config['num_tokens'],siglip_pooling=adapter_config['siglip_pooling']).to(dtype=torch.bfloat16)

            for name, p in self.transformer.named_parameters():
                p.original_name = name
                # 冻结基座
                p.requires_grad_(False)

            if is_main_process():
                print('[BasePipeline] IP-Adapter mode: NOT wrapping transformer with LoRA. '
                      'Make sure your IP-Adapter modules (e.g., proj head / attn processors) '
                      'are created separately and their parameters are passed to the optimizer.')

        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')


    def save_adapter(self, save_dir, peft_state_dict):
        raise NotImplementedError()

    def load_adapter_weights(self, adapter_path):
        # path = Path(adapter_path) / "mlp_clip2added_kv.pt"

        # dim = 1536                 # 例子：cross_attention_dim
        # id_embeddings_dim = 1152   # 例子：SigLIP 输出维度
        # num_tokens = 16            # 例子：你当时设置的 token 数

        # mlp = MLP_Clip2Added_kv(dim=dim, id_embeddings_dim=id_embeddings_dim, num_tokens=num_tokens)

        # state = torch.load(ckpt_path, map_location="cpu")
        # load_info = mlp.load_state_dict(state, strict=True)   # 不想严格就 strict=False
        # print(getattr(load_info, "missing_keys", []), getattr(load_info, "unexpected_keys", []))
        if is_main_process():
            print(f'Loading adapter weights from path {adapter_path}')
        safetensors_files = list(Path(adapter_path).glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise RuntimeError(f'No safetensors file found in {adapter_path}')
        if len(safetensors_files) > 1:
            raise RuntimeError(f'Multiple safetensors files found in {adapter_path}')
        adapter_state_dict = safetensors.torch.load_file(safetensors_files[0])
        modified_state_dict = {}
        model_parameters = set(name for name, p in self.transformer.named_parameters())
        for k, v in adapter_state_dict.items():
            # Replace Diffusers or ComfyUI prefix
            k = re.sub(r'^(transformer|diffusion_model)\.', '', k)
            # Replace weight at end for LoRA format
            k = re.sub(r'\.weight$', '.default.weight', k)
            if k not in model_parameters:
                raise RuntimeError(f'modified_state_dict key {k} is not in the model parameters')
            modified_state_dict[k] = v
        self.transformer.load_state_dict(modified_state_dict, strict=False)

    def load_and_fuse_adapter(self, path):
        peft_config = peft.LoraConfig.from_pretrained(path)
        lora_model = peft.get_peft_model(self.transformer, peft_config)
        self.load_adapter_weights(path)
        lora_model.merge_and_unload()

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False)

    def get_call_vae_fn(self, vae):
        raise NotImplementedError()

    def get_call_text_encoder_fn(self, text_encoder):
        raise NotImplementedError()

    def prepare_inputs(self, inputs, timestep_quantile=None):
        raise NotImplementedError()

    def to_layers(self):
        raise NotImplementedError()

    def model_specific_dataset_config_validation(self, dataset_config):
        pass

    # Get param groups that will be passed into the optimizer. Models can override this, e.g. SDXL
    # supports separate learning rates for unet and text encoders.
    def get_param_groups(self, parameters):
        return [{'params': parameters}]

    # Default loss_fn. MSE between output and target, with mask support.
    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                if 'pseudo_huber_c' in self.config:
                    c = self.config['pseudo_huber_c']
                    loss = torch.sqrt((output-target)**2 + c**2) - c
                else:
                    loss = F.mse_loss(output, target, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()
            return loss
        return loss_fn

    def enable_block_swap(self, blocks_to_swap):
        raise NotImplementedError('Block swapping is not implemented for this model')

    def prepare_block_swap_training(self):
        pass

    def prepare_block_swap_inference(self, disable_block_swap=False):
        pass
