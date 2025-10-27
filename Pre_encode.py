#!/usr/bin/env python3
import os
import shutil
from typing import List, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist

from transformers import AutoImageProcessor, SiglipVisionModel


# --------------------- DDP helpers ---------------------
def _ddp_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def _ddp_init():
    if _ddp_enabled():
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def _broadcast_obj(obj, src=0):
    if not _ddp_enabled():
        return obj
    container = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(container, src=src)
    return container[0]

def _barrier():
    if _ddp_enabled():
        dist.barrier()


# --------------------- small utils ---------------------
def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _npy_path(png_path: str, with_p: bool) -> str:
    """
    返回目标 .npy 路径：
      - with_p=True  -> {name}_wP.npy
      - with_p=False -> {name}.npy
    """
    s, _ = os.path.splitext(png_path)
    return s + ("_wP.npy" if with_p else ".npy")

def _move_png_and_txt(src_png: str, dest_dir: Optional[str], rank: int):
    """把 {name}.png 和 {name}.txt 移动到 dest_dir（覆盖同名）。"""
    if not dest_dir:
        print(f"[rank{rank}] [warn] fail_folder not set, skip moving: {src_png}")
        return
    os.makedirs(dest_dir, exist_ok=True)
    name = _stem(src_png)
    for ext in (".png", ".txt"):
        s = os.path.join(os.path.dirname(src_png), name + ext)
        d = os.path.join(dest_dir, name + ext)
        if os.path.exists(s):
            try:
                if os.path.exists(d):
                    os.remove(d)
                shutil.move(s, d)
            except Exception as e:
                print(f"[rank{rank}] [warn] move failed: {s} -> {d} ({e})")


# --------------------- Core ---------------------
def encode_pngs_to_npy(
    folder: str,
    model_name: str = "/home/zhengtianyu/yaowangzi/siglip-so400m-patch14-384/",
    batch_size: int = 256,
    overwrite: bool = False,
    fail_folder: Optional[str] = None,   # 因失败而迁移的目录 b
    include_cls: bool = False,           # with_p=False 时：是否保留 CLS（默认 False：丢弃 CLS）
    with_p: bool = False,                # True：保存池化后的全局向量 -> {name}_wP.npy；False：保存未池化 tokens -> {name}.npy
) -> int:
    """
    多卡并行编码 PNG -> 同名 NPY（写回 src）。

    - with_p=False（默认）：保存 **未池化**的特征序列：
        使用 SigLIP 的 `last_hidden_state` (B, T, D)，默认丢弃 CLS → (B, T-1, D)
        文件名：{name}.npy
        形状： (T-1, D) 或 (T, D)（当 include_cls=True）

    - with_p=True：保存 **池化后的全局向量**：
        优先取 `out.pooler_output`（B, D）；如无，则退化为 CLS（tokens[:, 0, :]）；
        若也不可用，则对 patch tokens 求均值。
        文件名：{name}_wP.npy
        形状： (D,)

    - 跳过已有目标文件（除非 overwrite=True）
    - 失败则把 {name}.png 和 {name}.txt 移到 fail_folder
    - 用 torchrun 启动多进程（cuda:0..N）
    """
    folder = os.path.abspath(folder)
    assert os.path.isdir(folder), f"Not a directory: {folder}"

    rank, world_size, local_rank = _ddp_init()

    # rank0 收集 .png 列表 + 生成待处理清单
    to_process: List[str] = []
    if rank == 0:
        all_pngs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png")]
        all_pngs.sort()
        # 针对目标产物名决定是否跳过
        if overwrite:
            to_process = all_pngs
        else:
            to_process = [p for p in all_pngs if not os.path.exists(_npy_path(p, with_p))]
        print(f"[rank0] total PNG: {len(all_pngs)}, to encode: {len(to_process)}, world_size={world_size}, with_p={with_p}")

    # 广播清单，切分任务
    to_process = _broadcast_obj(to_process, src=0)
    if len(to_process) == 0:
        return 0

    shard = to_process[rank::world_size]  # 均匀切片
    print(f"[rank{rank}] assigned {len(shard)} files.")

    # 设备 & 模型
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    image_encoder = SiglipVisionModel.from_pretrained(model_name)
    image_encoder.requires_grad_(False).eval().to(device)
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    # 编码
    written_local = 0
    for i in tqdm(range(0, len(shard), batch_size),
                  desc=f"r{rank}@{device}", unit="batch", position=rank, leave=True, dynamic_ncols=True):
        batch_files = shard[i:i + batch_size]

        # 读图 -> RGB
        images, paths_ok = [], []
        failed_open = []
        for p in batch_files:
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
                    paths_ok.append(p)
            except Exception as e:
                print(f"[rank{rank}] [warn] open failed: {p} ({e})")
                failed_open.append(p)

        # 先处理打不开的
        for p in failed_open:
            _move_png_and_txt(p, fail_folder, rank)

        if not images:
            continue

        try:
            # 预处理 & 前向
            inputs = image_processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

            with torch.no_grad():
                out = image_encoder(pixel_values=pixel_values)
                tokens = out.last_hidden_state  # (B, T, D), 通常 T = 1 + #patches（含 CLS）

                if with_p:
                    # 池化后的全局向量优先使用官方输出
                    pooled = getattr(out, "pooler_output", None)
                    if pooled is None:
                        print("what the fuck, no pooler_output?")
                        # 退化策略：优先 CLS，其次对 patch tokens 求均值
                        if tokens.shape[1] >= 1:
                            # CLS
                            pooled = tokens[:, 0, :]
                        else:
                            # 理论上不会发生；兜底均值
                            pooled = tokens.mean(dim=1)
                    feats_np = pooled.detach().cpu().numpy().astype(np.float32)  # (B, D)
                    # 保存一图一向量
                    for p, vec in zip(paths_ok, feats_np):
                        npy_path = _npy_path(p, with_p=True)
                        try:
                            if (not overwrite) and os.path.exists(npy_path):
                                continue
                            np.save(npy_path, vec, allow_pickle=False)
                            written_local += 1
                        except Exception as e:
                            print(f"[rank{rank}] [warn] save failed: {p} ({e})")
                            _move_png_and_txt(p, fail_folder, rank)
                else:
                    # 未池化：默认丢 CLS，只保留 patch tokens
                    if not include_cls and tokens.shape[1] >= 2:
                        tokens = tokens[:, 1:, :]
                    tokens_np = tokens.detach().cpu().numpy().astype(np.float32)  # (B, T' , D)
                    for p, tok in zip(paths_ok, tokens_np):
                        npy_path = _npy_path(p, with_p=False)
                        try:
                            if (not overwrite) and os.path.exists(npy_path):
                                continue
                            np.save(npy_path, tok, allow_pickle=False)
                            written_local += 1
                        except Exception as e:
                            print(f"[rank{rank}] [warn] save failed: {p} ({e})")
                            _move_png_and_txt(p, fail_folder, rank)

        except Exception as e_batch:
            # 整个 batch 失败（如 OOM）：把本 batch 的所有文件都移到 b
            print(f"[rank{rank}] [batch-skip] batch {i//batch_size} failed: {e_batch}")
            for p in batch_files:
                _move_png_and_txt(p, fail_folder, rank)
            continue

    # 汇总
    if _ddp_enabled():
        t = torch.tensor([written_local], device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        written_global = int(t.item())
    else:
        written_global = written_local

    if rank == 0:
        print(f"[rank0] done. global written={written_global}")
    _barrier()
    return written_global if rank == 0 else written_local


# --------------------- CLI ---------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str,
                    default="/home/zhengtianyu/yaowangzi/laion_1_1_sample", help="源目录（只找 .png）")
    ap.add_argument("--model_name", type=str,
                    default="/home/zhengtianyu/yaowangzi/siglip-so400m-patch14-384/")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true", help="重算目标产物（会覆盖已有 .npy）")
    ap.add_argument("--fail_folder", type=str, default=None, help="失败样本搬运目录 b")
    ap.add_argument("--include_cls", action="store_true",
                    help="with_p=False 时，保存未池化 tokens 是否保留 CLS（默认丢弃 CLS）")
    ap.add_argument("--with_p", action="store_true",
                    help="保存池化后的全局向量（{name}_wP.npy）；默认保存未池化 tokens（{name}.npy）")
    args = ap.parse_args()

    encode_pngs_to_npy(
        folder=args.folder,
        model_name=args.model_name,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        fail_folder=args.fail_folder,
        include_cls=args.include_cls,
        with_p=args.with_p,
    )
