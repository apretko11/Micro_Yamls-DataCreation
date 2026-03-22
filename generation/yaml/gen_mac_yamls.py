import json
from pathlib import Path

EXAMPLES_DIR = Path("examples")
DATA_INFO = Path("data/dataset_info.json")

# bases for mac datasets
BASES = [
    "x86_to_armv8mac",
    "armv8mac_to_x86",
    "armv8mac_to_riscv",
    "riscv_to_armv8mac",
]

SIZES = {
    "0p5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "1p5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "3p0b": "Qwen/Qwen2.5-Coder-3B-Instruct",
}

# training knobs: keep consistent with your current "full" configs
COMMON = {
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "template": "qwen",
    "cutoff_len": 8192,
    "packing": True,
    "max_samples": 5000000,
    "overwrite_cache": False,
    "preprocessing_num_workers": 2,
    "logging_steps": 10,
    "save_steps": 500,
    "plot_loss": True,
    "save_total_limit": 1,
    "save_only_model": True,
    "overwrite_output_dir": True,
    "push_to_hub": False,
    "hub_always_push": False,
    "report_to": "none",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": False,
    "learning_rate": 2.0e-5,
    "num_train_epochs": 0.5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "bf16": True,
    "ddp_timeout": 180000000,
    "flash_attn": "fa2",
    "enable_liger_kernel": True,
}

def yaml_dump_list(items, indent=2):
    out = []
    for x in items:
        out.append(" " * indent + f"- {x}")
    return "\n".join(out)

def main():
    info = json.load(open(DATA_INFO, "r", encoding="utf-8"))

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    created = []
    for base in BASES:
        shard_keys = sorted([k for k in info.keys() if k.startswith(base + "_")])
        if not shard_keys:
            raise SystemExit(f"No dataset_info keys found for base={base}. Did you generate shards?")
        for sz_tag, model_id in SIZES.items():
            fname = EXAMPLES_DIR / f"{base}_qwen25coder_{sz_tag}_full.yaml"
            hub_id = f"{base}_qwen25coder_{sz_tag}_full"
            out_dir = f"saves/{hub_id}"

            text = []
            text.append("### model")
            text.append(f"model_name_or_path: {model_id}")
            text.append(f"hub_model_id: {hub_id}")
            text.append("hub_strategy: end")
            text.append("#cache_dir: ~/.cache/huggingface\n")

            text.append("### method")
            text.append(f"stage: {COMMON['stage']}")
            text.append(f"do_train: {str(COMMON['do_train']).lower()}")
            text.append(f"finetuning_type: {COMMON['finetuning_type']}\n")

            text.append("### dataset")
            text.append("dataset:")
            text.append(yaml_dump_list(shard_keys, indent=2))
            text.append(f"\ntemplate: {COMMON['template']}")
            text.append(f"cutoff_len: {COMMON['cutoff_len']}")
            text.append(f"packing: {str(COMMON['packing']).lower()}")
            text.append(f"max_samples: {COMMON['max_samples']}")
            text.append(f"overwrite_cache: {str(COMMON['overwrite_cache']).lower()}")
            text.append(f"preprocessing_num_workers: {COMMON['preprocessing_num_workers']}\n")

            text.append("### output")
            text.append(f"output_dir: {out_dir}")
            text.append(f"logging_steps: {COMMON['logging_steps']}")
            text.append(f"save_steps: {COMMON['save_steps']}")
            text.append(f"plot_loss: {str(COMMON['plot_loss']).lower()}")
            text.append(f"save_total_limit: {COMMON['save_total_limit']}")
            text.append(f"save_only_model: {str(COMMON['save_only_model']).lower()}")
            text.append(f"overwrite_output_dir: {str(COMMON['overwrite_output_dir']).lower()}")
            text.append(f"push_to_hub: {str(COMMON['push_to_hub']).lower()}")
            text.append("hub_always_push: false")
            text.append(f"report_to: {COMMON['report_to']}\n")

            text.append("### train")
            text.append(f"per_device_train_batch_size: {COMMON['per_device_train_batch_size']}")
            text.append(f"gradient_accumulation_steps: {COMMON['gradient_accumulation_steps']}")
            text.append(f"gradient_checkpointing: {str(COMMON['gradient_checkpointing']).lower()}")
            text.append(f"learning_rate: {COMMON['learning_rate']}")
            text.append(f"num_train_epochs: {COMMON['num_train_epochs']}")
            text.append(f"lr_scheduler_type: {COMMON['lr_scheduler_type']}")
            text.append(f"warmup_ratio: {COMMON['warmup_ratio']}")
            text.append(f"bf16: {str(COMMON['bf16']).lower()}")
            text.append(f"ddp_timeout: {COMMON['ddp_timeout']}")
            text.append(f"flash_attn: {COMMON['flash_attn']}")
            text.append(f"enable_liger_kernel: {str(COMMON['enable_liger_kernel']).lower()}")
            text.append("")

            fname.write_text("\n".join(text), encoding="utf-8")
            created.append(str(fname))

    print("Created YAMLs:")
    for c in created:
        print(" ", c)

if __name__ == "__main__":
    main()