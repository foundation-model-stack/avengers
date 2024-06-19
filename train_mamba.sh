#!/usr/bin/bash

git clone -b mamba2 https://github.com/foundation-model-stack/fms-fsdp.git
cd fms-fsdp

# wrap below with torchrun in your env
main_training.py
  --model_variant=mamba_2.9b
  --use_dummy_dataset=False
  --data_path=/gpfs
  --datasets=fineweb-edu
  --weights=1
  --seq_length=4096
  --vocab_size=128256
  --logical_shards=960
  --ckpt_load_path=/gpfs/lchu/ckpt/mamba2_3b
  --ckpt_save_path=/gpfs/lchu/ckpt/mamba2_3b
  --fsdp_activation_checkpointing=False
  --selective_checkpointing=1
  --sharding_strategy=fsdp
  --low_cpu_fsdp=False
  --batch_size=4
  --num_steps=100000
  --learning_rate=4.8e-4
  --report_interval=100
  --checkpoint_interval=100000
  --use_profiler=False
  --use_torch_compile=False
  --tracker=None
  --tracker_dir=/aim/aim
  --tracker_project_name=lchu-mamba2-3b
  --tracker_run_id=None
