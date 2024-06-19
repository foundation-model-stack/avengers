#!/usr/bin/bash

git clone https://github.com/foundation-model-stack/fms-fsdp.git
cd fms-fsdp

# wrap below with torchrun in your env
main_training.py
  --model_variant=llama3_1.8b_4k
  --use_dummy_dataset=False
  --data_path=/gpfs
  --datasets=fineweb-edu
  --weights=1
  --seq_length=4096
  --vocab_size=128256
  --bos_token=None
  --eos_token=128000
  --logical_shards=960
  --ckpt_load_path=/gpfs/lchu/ckpt/llama3_8k_bos_new_4k
  --ckpt_save_path=/gpfs/lchu/ckpt/llama3_8k_bos_new_4k
  --fsdp_activation_checkpointing=False
  --selective_checkpointing=1
  --sharding_strategy=hsdp
  --low_cpu_fsdp=False
  --batch_size=4
  --num_steps=33000
  --learning_rate=6e-4
  --report_interval=100
  --checkpoint_interval=33000
  --use_profiler=False
  --use_torch_compile=True
  --tracker=None
  --tracker_dir=/aim/aim
  --tracker_project_name=lchu-ablation
  --tracker_run_id=None
