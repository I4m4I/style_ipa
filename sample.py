import argparse
import os
import wandb
from datetime import datetime, timezone
import shutil
import glob
import time
import random
import json
import inspect
from pathlib import Path
from collections import defaultdict

import toml
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocess as mp
import numpy as np

from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP, empty_cuda_cache
import utils.saver
from utils.isolate_rng import isolate_rng
from utils.patches import apply_patches
from utils.unsloth_utils import unsloth_checkpoint
from utils.pipeline import ManualPipelineModule

# --- IP-Adapter / SigLIP 依赖 ---
from transformers import SiglipVisionModel, AutoImageProcessor
from torchvision.transforms.functional import to_pil_image
import math
from typing import Iterable, List, Tuple, Optional
from PIL import Image
from transformers import SiglipVisionModel, AutoImageProcessor


# needed for broadcasting Queue in dataset.py
mp.current_process().authkey = b'afsaskgfdjh4'

wandb_enable = False

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--reset_dataloader', action='store_true', help='Start dataloader from scratch when resuming from checkpoint, i.e. only load the optimizer states.')
parser.add_argument('--regenerate_cache', action='store_true', help='Force regenerate cache.')
parser.add_argument('--cache_only', action='store_true', help='Cache model inputs then exit.')
parser.add_argument('--trust_cache', action='store_true', help='Load from metadata cache files if they exist, without checking if any fingerprints have changed. Can make loading much faster for large datasets.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true', help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
parser.add_argument('--dump_dataset', type=Path, default=None, help='Decode cached latents and dump the dataset to this directory.')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.state = defaultdict(dict)
        self.param_groups = []

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none: bool = True):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config or 'save_every_n_steps' in config or 'save_every_n_examples' in config

    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('reentrant_activation_checkpointing', False)
    if config['activation_checkpointing'] == 'unsloth':
        config['reentrant_activation_checkpointing'] = True
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get('transformer_dtype', None):
        model_config['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if adapter_config['type'] == 'lora':
            if 'alpha' in adapter_config:
                raise NotImplementedError(
                    'This script forces alpha=rank to make the saved LoRA format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
                )
            adapter_config['alpha'] = adapter_config['rank']
            adapter_config.setdefault('dropout', 0.0)
            adapter_config.setdefault('dtype', model_dtype_str)
            adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]
        elif adapter_type == 'ip-adapter':  # --- 新增: 支持 IPA ---
            # 这些默认值可在 TOML 里覆盖
            adapter_config.setdefault('siglip_model_name', 'google/siglip-so400m-patch14-384')
            adapter_config.setdefault('num_tokens', 128)
            adapter_config.setdefault('ref_image_key', 'con-image')      # 参考图字段名，见下文 transform
            adapter_config.setdefault('drop_prob', 0.05)                  # 随机丢弃图像引导的比例
            # IPA 不需要 lora 的 dtype/alpha 等
        else:
            pass

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_every_n_examples', None)
    config.setdefault('eval_before_first_step', True)
    config.setdefault('compile', False)
    config.setdefault('x_axis_examples', False)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


# Need to preload all micro batches since pulling from the dataloader does IPC between the
# first and last stage. Can't do that during the train or inference pipeline schedule execution
# because it conflicts with the send / recv steps.
def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    num_micro_batches = num_micro_batches or engine.micro_batches
    if not (engine.is_first_stage() or engine.is_last_stage()):
        return None
    dataloader_iter = iter(dataloader)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.set_eval_quantile(quantile)
    total_loss = 0
    count = 0
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(eval_dataloader, model_engine, num_micro_batches=eval_gradient_accumulation_steps)
        loss = model_engine.eval_batch(iterator, num_micro_batches=eval_gradient_accumulation_steps).item()
        eval_dataloader.sync_epoch()
        if pbar:
            pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break

    eval_dataloader.reset()
    return total_loss / count


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    pbar_total = 0
    for eval_dataloader in eval_dataloaders.values():
        pbar_total += len(eval_dataloader) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps
    if is_main_process():
        print('Running eval')
        pbar = tqdm(total=pbar_total)
    else:
        pbar = None

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=pbar)
            losses.append(loss)
            if is_main_process():
                tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
                if wandb_enable:
                    wandb.log({f'{name}/loss_quantile_{quantile:.2f}': loss, 'step': step})
        avg_loss = sum(losses) / len(losses)
        if is_main_process():
            tb_writer.add_scalar(f'{name}/loss', avg_loss, step)
            if wandb_enable:
                wandb.log({f'{name}/loss': avg_loss, 'step': step})

    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)
        if wandb_enable:
            wandb.log({'eval/eval_time_sec': duration, 'step': step})
        pbar.close()


def evaluate(model, model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps, disable_block_swap):
    if len(eval_dataloaders) == 0:
        return
    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    with torch.no_grad(), isolate_rng():
        seed = get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)
    empty_cuda_cache()
    model.prepare_block_swap_training()


def distributed_init(args):
    """Initialize distributed training environment."""
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = args.local_rank

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(args.master_port)

    return world_size, rank, local_rank


def get_prodigy_d(optimizer):
    d = 0
    for group in optimizer.param_groups:
        d += group['d']
    return d / len(optimizer.param_groups)


def _get_automagic_lrs(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            lr = optimizer._get_lr(group, state)
            lrs.append(lr)
    lrs = torch.stack(lrs)
    return lrs, lrs.mean()

# new added siglip
def encode_images_to_npy(
    folder: str,
    model_name: str = "/home/zhengtianyu/yaowangzi/siglip-so400m-patch14-384/",
    batch_size: int = 32,
    device: Optional[str] = None,
    overwrite: bool = False,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"),
) -> int:
    """
    扫描 `folder` 下的所有图片，若对应的 .npy 不存在则用 SigLIP 视觉编码器提取全局特征并保存为同名 .npy。
    - 若所有图片都已存在 .npy（且 overwrite=False），则跳过模型加载与处理。
    - 返回本次实际新写入的 .npy 文件数量。
    """
    folder = os.path.abspath(folder)
    assert os.path.isdir(folder), f"Not a directory: {folder}"

    # 收集所有图片（不递归）
    img_files: List[str] = []
    for name in os.listdir(folder):
        lower = name.lower()
        if any(lower.endswith(ext) for ext in extensions):
            img_files.append(os.path.join(folder, name))
    img_files.sort()

    if not img_files:
        print(f"[encode_images_to_npy] No images found in: {folder}")
        return 0

    # 需要处理的图片：不存在同名 .npy 或者要求覆盖时
    def to_npy(p: str) -> str:
        stem, _ = os.path.splitext(p)
        return stem + ".npy"

    to_process = [p for p in img_files if overwrite or not os.path.exists(to_npy(p))]

    if len(to_process) == 0:
        print("[encode_images_to_npy] All images already processed. Nothing to do.")
        return 0

    # 只有确实需要处理时才加载模型与处理器
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[encode_images_to_npy] {len(img_files)} images found, "
          f"{len(to_process)} to process. Loading model on {device} ...")

    image_encoder = SiglipVisionModel.from_pretrained(model_name)
    image_encoder.requires_grad_(False).eval().to(device)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    print("[encode_images_to_npy] image encoder ok")

    written = 0
    # 批处理
    for i in tqdm(range(0, len(to_process), batch_size), desc="Encoding", unit="batch"):
        batch_files = to_process[i:i + batch_size]

        # 读取并转 RGB
        images = []
        for p in batch_files:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))

        # 预处理 -> pixel_values: (B, 3, H, W)
        inputs = image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

        with torch.no_grad():
            outputs = image_encoder(pixel_values=pixel_values)
            # SigLIP 有 pooler_output=全局向量（如 1152 维）；兜底用 mean-pooled token
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeds = outputs.pooler_output  # (B, D)
            else:
                embeds = outputs.last_hidden_state.mean(dim=1)  # (B, D)

        # 保存为同名 .npy；默认 float32
        embeds_np = embeds.detach().cpu().numpy().astype(np.float32)
        for p, vec in zip(batch_files, embeds_np):
            np.save(to_npy(p), vec, allow_pickle=False)
            written += 1

    print(f"[encode_images_to_npy] Done. Wrote {written} npy files.")
    return written

if __name__ == '__main__':
    apply_patches()

    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))

    set_config_defaults(config)
    common.AUTOCAST_DTYPE = config['model']['dtype']
    dataset_util.UNCOND_FRACTION = config.get('uncond_fraction', 0.0)
    if map_num_proc := config.get('map_num_proc', None):
        dataset_util.NUM_PROC = map_num_proc

    model_type = config['model']['type']
    from models.wan import wan
    model = wan.WanPipeline(config)
    model.load_diffusion_model()

    

    

