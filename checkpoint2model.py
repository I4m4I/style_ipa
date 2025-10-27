#!/usr/bin/env python3
import argparse, json, os, glob
from pathlib import Path

import toml
import torch
import deepspeed
from deepspeed import comm as dist

from utils.common import DTYPE_MAP
from utils.pipeline import ManualPipelineModule
import utils.saver  # Saver 用到
# 注意：不要 import train.py（它会 parse_args）；这里复刻所需最小逻辑

def set_config_defaults(config: dict):
    assert 'save_every_n_epochs' in config or 'save_every_n_steps' in config or 'save_every_n_examples' in config
    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('reentrant_activation_checkpointing', False)
    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('compile', False)
    config.setdefault('x_axis_examples', False)

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get('transformer_dtype', None):
        model_config['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)

    if 'adapter' in config:
        adapter = config['adapter']
        if adapter.get('type') == 'ip-adapter':
            adapter.setdefault('siglip_model_name', 'google/siglip-so400m-patch14-384')
            adapter.setdefault('num_tokens', 128)
            adapter.setdefault('ref_image_key', 'con-image')
            adapter.setdefault('drop_prob', 0.05)

def get_most_recent_run_dir(output_dir: str) -> str:
    # 和 train.py 一样：取 output_dir 下最新的子目录
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]

def distributed_init(args):
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(args.master_port)
    return world_size, rank, args.local_rank

class DummyLoader:
    """极简的占位 dataloader，不做任何数据工作。"""
    def __init__(self, epoch:int=1):
        self.epoch = epoch
    def state_dict(self):
        return {'epoch': self.epoch}
    def load_state_dict(self, sd):
        self.epoch = sd.get('epoch', self.epoch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to TOML config.')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, required=True,
                        help='Same semantics as train.py: True=latest, or specify a folder name')
    parser.add_argument('--master_port', type=int, default=29500)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 读 config，并按 train.py 的方式规范化
    with open(args.config) as f:
        # 与 train.py 一致：先 toml -> json roundtrip，避免不可 picklable 的 inline table
        config = json.loads(json.dumps(toml.load(f)))
    set_config_defaults(config)

    # 构建模型（只处理 wan；如果你配了其他类型，按需扩展分支）
    if config['model']['type'] != 'wan':
        raise NotImplementedError("当前脚本仅针对 model.type == 'wan'")
    from models.wan import wan as wan_mod  # 延迟 import
    model = wan_mod.WanPipeline(config)

    # 初始化分布式 & Deepspeed（与 train.py 同步）
    deepspeed.init_distributed()
    # 让每个 rank 绑到自己的 GPU
    torch.cuda.set_device(dist.get_rank())

    # 只加载模型，不做任何 dataset/cache
    model.load_diffusion_model()

    # 适配器（如配置了 IP-Adapter/Lora），完全复用 train.py 的逻辑
    is_adapter = True
    is_ip_adapter = False
    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
        is_adapter = True
        if adapter_config.get('type') == 'ip-adapter':
            is_ip_adapter = True
        if init_from_existing := adapter_config.get('init_from_existing', None):
            model.load_adapter_weights(init_from_existing)

    # 解析 run_dir（与 train.py 同语义：字符串/True=latest）
    if args.resume_from_checkpoint is True:
        run_dir = get_most_recent_run_dir(config['output_dir'])
    else:
        run_dir = os.path.join(config['output_dir'], args.resume_from_checkpoint)
        if not os.path.exists(run_dir):
            raise ValueError(f"Checkpoint directory {run_dir} does not exist")

    # 构建 layers -> ManualPipelineModule（与 train.py 一致）
    layers = model.to_layers()
    additional_kwargs = {}
    if config['activation_checkpointing']:
        import functools
        checkpoint_func = functools.partial(torch.utils.checkpoint.checkpoint,
                                            use_reentrant=config['reentrant_activation_checkpointing'])
        additional_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })

    pipeline_model = ManualPipelineModule(
        layers=layers,
        num_stages=config.get('pipeline_stages', 1),
        partition_method=config.get('partition_method', 'parameters'),
        manual_partition_split=config.get('partition_split', [len(layers) / config.get('pipeline_stages', 1)]),
        loss_fn=model.get_loss_fn(),
        **additional_kwargs
    )

    if config['compile']:
        pipeline_model.compile()

    # 最小 DS config（与 train.py 的字段名一致）
    ds_config = {
        'train_micro_batch_size_per_gpu': config.get('micro_batch_size_per_gpu', 1),
        'gradient_accumulation_steps':    config.get('gradient_accumulation_steps', 1),
        'gradient_clipping':              0.0,
        'steps_per_print':                config.get('steps_per_print', 1),
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        config=ds_config,
    )

    # （可选）通信 dtype 同步（train.py 里做了）
    communication_data_type = config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    # —— 重点：只加载 checkpoint，不做 dataloader/dataset ——
    load_path, client_state = model_engine.load_checkpoint(
        run_dir,
        load_module_strict=False,
        load_lr_scheduler_states=False,
    )
    if load_path is None:
        raise RuntimeError(f"Failed to load checkpoint from: {run_dir}")

    # 构造 Dummy dataloader，完全不读数据
    train_dataloader = DummyLoader(epoch=client_state.get('custom_loader', {}).get('epoch', 1))

    # Saver 的构造与 train.py 一致
    saver = utils.saver.Saver(
        args, config, is_adapter, is_ip_adapter, run_dir,
        model, train_dataloader, model_engine, pipeline_model
    )

    # 只执行一次 process_epoch；不给它制造“完成一整轮”的条件（避免任何保存/额外副作用）
    # 这里的 step/examples 与 train.py 的形状一致（只是很小的占位数）
    dp_world = model_engine.grid.get_data_parallel_world_size()
    global_bs = model_engine.train_micro_batch_size_per_gpu() * model_engine.gradient_accumulation_steps() * dp_world
    epoch = train_dataloader.epoch
    step = 1
    examples = global_bs

    new_epoch, checkpointed, saved = saver.process_epoch(2, step, examples)

    if dist.get_rank() == 0:
        print(f"[DONE] process_epoch called once. "
              f"returned: new_epoch={new_epoch}, checkpointed={checkpointed}, saved={saved}")

if __name__ == "__main__":
    main()

"""
deepspeed --num_gpus=1 /home/zhengtianyu/yaowangzi/diffusion-pipe/checkpoint2model.py \
  --deepspeed \
  --config /home/zhengtianyu/yaowangzi/diffusion-pipe/examples/wan5b.toml \
  --resume_from_checkpoint "20251007_16-10-42"
"""
