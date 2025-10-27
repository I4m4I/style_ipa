#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, argparse, glob, json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn

# 确保能 import wan.py 以及其相对依赖（建议在repo根目录运行）
sys.path.append(os.getcwd())

from models.wan.wan import WanPipeline  # 你上传的文件
from models.base import MLP_Clip2Added_kv  # 与训练同名的 MLP 类

# ---------------------------
# 工具函数
# ---------------------------

def str2dtype(s: str):
    s = s.lower()
    if s in ["bf16", "bfloat16", "bfloat"]:
        return torch.bfloat16
    if s in ["fp16", "float16", "half"]:
        return torch.float16
    if s in ["fp32", "float32", "full", "f32"]:
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")

def load_batch_items(a_dir: Path, names, is_pooling=True):
    """
    读取一个 batch：
      - captions: list[str]
      - id_embeds: torch.Tensor (B, D)
      - ref_imgs:  list[PIL.Image]
    """
    captions = []
    embeds = []
    refs = []
    for name in names:
        txt = (a_dir / f"{name}.txt").read_text(encoding="utf-8").strip()
        cap = txt if len(txt) > 0 else " "
        captions.append(cap)
        if is_pooling:
            npy = np.load(a_dir / f"{name}_wP.npy")
        else:
            npy = np.load(a_dir / f"{name}.npy")
        embeds.append(torch.from_numpy(npy).float())  # 先float32, 进入 MLP 再 autocast

        # 参考图用于并排可视化；进入模型的 SigLIP 已离线完成
        ref_img = Image.open(a_dir / f"{name}.png").convert("RGB")
        refs.append(ref_img)

    id_embeds = torch.stack(embeds, dim=0)  # (B, D)
    return captions, id_embeds, refs

def make_pairs_and_save(ref_imgs, gen_imgs, names, out_dir: Path, save_single=False):
    """
    ref_imgs: list[PIL.Image]
    gen_imgs: list[PIL.Image]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for r, g, name in zip(ref_imgs, gen_imgs, names):
        # 调整参考图到生成图尺寸，或反之；这里以生成图尺寸为准
        if r.size != g.size:
            r = r.resize(g.size, Image.BICUBIC)

        W, H = g.size
        pair = Image.new("RGB", (W * 2, H))
        pair.paste(r, (0, 0))
        pair.paste(g, (W, 0))
        pair.save(out_dir / f"{name}_pair.png")

        if save_single:
            g.save(out_dir / f"{name}.png")

def decode_to_pil(latents: torch.Tensor, vae) -> list:
    """
    latents: (B, 4, F=1, h, w)
    vae:    pipeline.vae (有 .model 和 .scale)
    return: list of PIL.Image
    """
    with torch.no_grad():
        img = vae.model.decode(latents, vae.scale).float().clamp_(-1, 1)  # (B, 3, 1, H, W)
        img = img[:, :, 0]  # 只保留第0帧 -> (B, 3, H, W)
        img = (img + 1) / 2.0
        img = (img * 255).round().to(torch.uint8).cpu().numpy()  # (B, 3, H, W)

    pils = []
    for arr in img:
        pil = Image.fromarray(np.moveaxis(arr, 0, -1))  # CHW -> HWC
        pils.append(pil)
    return pils

def build_layers_for_inference(pipeline: WanPipeline, mlp: nn.Module):
    """
    将 MLP 注入 pipeline，使 to_layers() 插入 IPAdapterMLPLayer，然后构造层序列。
    """
    pipeline.MLP_Clip2Added_kv = mlp
    layers = pipeline.to_layers()  # [IPAdapterMLPLayer?, InitialLayer, TransformerLayers..., FinalLayer]
    # 将子模块都丢到 cuda（offloader 为 dummy，不会互斥）
    for m in layers:
        m.to("cuda")
        m.eval()
    return layers

@torch.no_grad()
def predict_velocity_once(layers, x_t, t_in, text_embeddings, seq_lens, id_embeds, h_lat, w_lat, model_type="i2v_v2", is_adapter=True):
    """
    通过 [IPAdapterMLP] -> InitialLayer -> blocks -> FinalLayer 得到当前速度场 v(x_t, t)
    - x_t:         (B, 4, 1, h, w)
    - t_in:        (B,)   已乘 1000 的标量时间
    - text_*:      由 T5 得到（cache_text_embeddings=True 路径）
    - id_embeds:   (B, D) 直接来自 .npy
    - 返回:        (B, 4, 1, h, w)
    """
    B = x_t.size(0)
    device = x_t.device
    dtype  = x_t.dtype

    # Wan2.2（i2v_v2）无 CLIP 视觉编码，clip_fea 传空张量，InitialLayer 内部会置 None
    clip_fea = torch.empty(0, device=device, dtype=dtype)

    # i2v_v2 路径下，InitialLayer 期望有 y（用于 mask+拼接）；采样时传 0 即可
    y = torch.zeros(B, 4, 1, h_lat, w_lat, device=device, dtype=dtype)

    # 组装 7 元组（含 id_embeds），IPAdapterMLPLayer 会扩展成 8 元组 (+added_kv)
    if is_adapter:
        inputs = (x_t, y, t_in, text_embeddings, seq_lens, clip_fea, id_embeds)
    else:
        inputs = (x_t, y, t_in, text_embeddings, seq_lens, clip_fea)

    out = inputs
    for layer in layers:
        out = layer(out)
    v = out  # FinalLayer 直接返回 (B, 4, 1, h, w)
    return v

def heun_step(layers, x, t_cur, t_next, text_embeddings, seq_lens, id_embeds, h_lat, w_lat, is_adapter=True):
    """
    Heun 二阶：x_{k+1} = x_k - h * 0.5*(f_k + f_{k+1})
    t 输入给模型需乘 1000
    """
    h = (t_cur - t_next)  # 正数
    t_in_cur  = torch.full((x.size(0),), t_cur  * 1000.0, device=x.device, dtype=torch.float32)
    v_cur = predict_velocity_once(layers, x, t_in_cur, text_embeddings, seq_lens, id_embeds, h_lat, w_lat, is_adapter=is_adapter)

    x_euler = x - h * v_cur
    t_in_next = torch.full((x.size(0),), t_next * 1000.0, device=x.device, dtype=torch.float32)
    v_next = predict_velocity_once(layers, x_euler, t_in_next, text_embeddings, seq_lens, id_embeds, h_lat, w_lat, is_adapter=is_adapter)

    x_next = x - h * 0.5 * (v_cur + v_next)
    return x_next

def euler_step(layers, x, t_cur, t_next, text_embeddings, seq_lens, id_embeds, h_lat, w_lat, is_adapter=True):
    h = (t_cur - t_next)
    t_in_cur = torch.full((x.size(0),), t_cur * 1000.0, device=x.device, dtype=torch.float32)
    v_cur = predict_velocity_once(layers, x, t_in_cur, text_embeddings, seq_lens, id_embeds, h_lat, w_lat, is_adapter=True)
    x_next = x - h * v_cur
    return x_next

# ---------------------------
# 主函数
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Sample images with Wan2.2 + IP-Adapter (MLP) using RF ODE")
    parser.add_argument("--a_dir", default='/home/zhengtianyu/yaowangzi/laion_1_1_sample1', type=str, help="包含 {name}.png/.npy/.txt 的文件夹")
    parser.add_argument("--out_dir", default='/home/zhengtianyu/yaowangzi/diffusion-pipep/sample5_60k', type=str, help="输出文件夹（保存 并排图/可选生成图）")

    parser.add_argument("--wan_ckpt_dir",default="/home/zhengtianyu/yaowangzi/Wan2.2-TI2V-5B", type=str, help="Wan2.2 顶层 checkpoint 目录（含 t5/vae 等）")
    # parser.add_argument("--transformer_path", type=str, required=True, help="transformer safetensors（文件或目录）")

    parser.add_argument("--mlp_path",default="/home/zhengtianyu/yaowangzi/diffusion-pipe/data/diffusion_pipe_training_runs/wp/20251013_05-37-24/step62900/mlp_clip2added_kv.pt", type=str, help="MLP_Clip2Added_kv 的 .pt（你训练得到的 state_dict）")
    parser.add_argument("--id_dim", type=int, default=1152, help="SigLIP 向量维度（so400m-384 通常 1152）")
    parser.add_argument("--num_tokens", type=int, default=128, help="IP-Adapter 生成的 tokens 数（训练时设定）")

    parser.add_argument("--width",  type=int, default=512, help="最终图像宽（像素）")
    parser.add_argument("--height", type=int, default=512, help="最终图像高（像素）")
    parser.add_argument("--vae_downsample", type=int, default=8, help="VAE 下采样倍率（WAN2.1/2.2 通常是 8）")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30, help="RF 采样步数")
    parser.add_argument("--method", type=str, default="heun", choices=["heun", "euler"])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--save_single", action="store_true", help="除了并排，再单独保存生成图")

    parser.add_argument("--is_ipa", default=False, type=bool)
    parser.add_argument("--is_pooling", default=True, type=bool)

    args = parser.parse_args()

    a_dir = Path(args.a_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype  = str2dtype(args.dtype)

    # ----------------- 构建 WanPipeline 并加载权重 -----------------
    config = {
        "model": {
            "ckpt_path": args.wan_ckpt_dir,
            # "transformer_path": args.transformer_path,
            "dtype": dtype,
            "cache_text_embeddings": True,
            # 和训练一致的 t 分布裁剪；默认 [0,1] 全域
            "min_t": 0.0,
            "max_t": 1.0,
        },
        "adapter": {
            "type": "ip-adapter",
        },
        # 禁用重入检查点等，尽量简单
        "reentrant_activation_checkpointing": False,
    }

    print(">> init WanPipeline ...")
    pipeline = WanPipeline(config)  # 解析 config.json、加载 T5/CLIP/VAE 基座（在 CPU）
    pipeline.load_diffusion_model() # 延迟加载 transformer（在 CPU）  :contentReference[oaicite:6]{index=6}
    # 将核心模块转到 CUDA
    pipeline.transformer.to(device).train(False)
    pipeline.text_encoder.model.to(device).eval()
    pipeline.vae.model.to(device).eval()

    # ----- 构建并加载 MLP（根据 transformer 的 dim 自动设置） -----
    # 取主干维度（1536 for 1.3B / 5120 for 14B）
    cross_dim = int(pipeline.transformer.dim) if hasattr(pipeline.transformer, "dim") else pipeline.json_config.get("dim", 5120)
    print(f">> IP-Adapter MLP dims: cross_dim={cross_dim}, id_dim={args.id_dim}, num_tokens={args.num_tokens}")

    mlp = MLP_Clip2Added_kv(dim=cross_dim, id_embeddings_dim=args.id_dim, num_tokens=args.num_tokens,siglip_pooling=args.is_pooling)
    mlp.load_state_dict(torch.load(args.mlp_path, map_location="cpu"), strict=True)
    mlp.to(device).eval()

    # 组装层序列（会把 IPAdapterMLPLayer 插在最前）
    layers = build_layers_for_inference(pipeline, mlp)  # 参考 to_layers 定义  :contentReference[oaicite:7]{index=7}

    # 文本编码函数（T5）
    call_text = pipeline.get_call_text_encoder_fn(pipeline.text_encoder.model)  # 返回 text_embeddings + seq_lens  :contentReference[oaicite:8]{index=8}

    # ----------------- 数据准备 -----------------
    # 扫描 a_dir 内部的样本名（必须三件套齐全）
    names = []
    for p in sorted(a_dir.glob("*.npy")):
        base = p.stem
        if (a_dir / f"{base}.txt").exists() and (a_dir / f"{base}.png").exists():
            names.append(base)
    assert len(names) > 0, f"No valid triplets found in {a_dir}"

    # 输出尺寸 → latent 尺寸
    H_img, W_img = args.height, args.width
    h_lat, w_lat = H_img // args.vae_downsample, W_img // args.vae_downsample

    step_fn = heun_step if args.method == "heun" else euler_step

    # ----------------- 批量采样 -----------------
    B = args.batch_size
    ts = torch.linspace(1.0, 0.0, steps=args.steps + 1, device=device)  # 归一化时间（给模型时乘1000）

    for i in range(0, len(names), B):
        chunk = names[i : i + B]
        print(f">> Sampling batch {i//B+1} / {math.ceil(len(names)/B)}  (size={len(chunk)})")

        captions, id_embeds, ref_imgs = load_batch_items(a_dir, chunk, is_pooling=args.is_pooling)
        id_embeds = id_embeds.to(device)
        # if args.is_ipa is False:
        #     id_embeds = torch.zeros_like(id_embeds)
        # T5 文本嵌入（cache_text_embeddings=True 路径）
        te = call_text(captions, is_video=False)
        text_embeddings = te["text_embeddings"].to(device)
        seq_lens = te["seq_lens"].to(device)

        # x 初始化为高斯噪声 latent；一帧 (F=1)
        x = torch.randn(len(chunk), 48, 1, 64, 64, device=device, dtype=dtype)


        # RF 迭代
        for k in range(args.steps):
            if k % 10 == 0 or k == args.steps - 1:
                print(k)
            t_cur, t_next = ts[k].item(), ts[k+1].item()
            x = step_fn(layers, x, t_cur, t_next, text_embeddings, seq_lens, id_embeds, h_lat, w_lat, is_adapter=args.is_ipa)

        # 解码并保存
        gen_pils = decode_to_pil(x.to(pipeline.vae.dtype), pipeline.vae)
        make_pairs_and_save(ref_imgs, gen_pils, chunk, out_dir, save_single=args.save_single)

    print("Done.")

if __name__ == "__main__":
    main()

# python sampler.py