import argparse, json, os
from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="Stream HF dataset -> ShareGPT JSON shards + dataset_info.json (per-shard keys)")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--dataset-name", required=True, help="Base dataset name, e.g. x86_to_armv8mac")
    p.add_argument("--src-col", required=True)
    p.add_argument("--tgt-col", required=True)

    p.add_argument("--max-context-length", type=int, default=8192)
    p.add_argument("--context-length-offset", type=int, default=1000)

    p.add_argument("--system-prompt", default="You are a helpful coding assistant specialized in converting assembly.")
    p.add_argument("--user-prompt", default="Convert the following assembly code to the target ISA:\n```asm\n{asm}\n```")
    p.add_argument("--output-prompt", default="```asm\n{asm}\n```")

    p.add_argument("--data-dir", default="data")
    p.add_argument("--shard-size", type=int, default=150000, help="Examples per shard (default 150k)")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    dataset_info_path = os.path.join(args.data_dir, "dataset_info.json")
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    print(f"Streaming dataset: {args.dataset_id} split={args.split}")
    ds = load_dataset(args.dataset_id, split=args.split, streaming=True)

    shard_idx = 0
    buf = []
    total_written = 0

    def flush():
        nonlocal shard_idx, buf, dataset_info, total_written
        if not buf:
            return
        out_fn = f"{args.dataset_name}_part_{shard_idx:03d}.json"
        out_path = os.path.join(args.data_dir, out_fn)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(buf, f, ensure_ascii=False)

        key = f"{args.dataset_name}_{shard_idx:03d}"
        dataset_info[key] = {
            "file_name": out_fn,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }

        total_written += len(buf)
        print(f"Wrote {out_path} with {len(buf)} examples (total={total_written})")
        buf = []
        shard_idx += 1

    for example in tqdm(ds, desc=f"Processing {args.dataset_name}"):
        src = example.get(args.src_col)
        tgt = example.get(args.tgt_col)
        if not src or not tgt:
            continue
        if len(src) + len(tgt) + args.context_length_offset > args.max_context_length:
            continue

        entry = {
            "conversations": [
                {"role": "user", "content": args.user_prompt.format(asm=src)},
                {"role": "assistant", "content": args.output_prompt.format(asm=tgt)},
            ],
            "system": args.system_prompt,
        }
        buf.append(entry)
        if len(buf) >= args.shard_size:
            flush()

    flush()

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"Updated {dataset_info_path}")
    print("Done.")

if __name__ == "__main__":
    main()