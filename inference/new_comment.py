import time
import pandas as pd
from datasets import load_dataset
from openai import OpenAI, NotFoundError

# ==============================
# CONFIGURATION
# ==============================
HF_DATASET = "ahmedheakl/gg-bench-armv8-O2"
OUTPUT_PATH = "ggbench-armv8-O2-commented.parquet"
CHECKPOINT_INTERVAL = 25

MODEL_PRIMARY = "gpt-5"
MODEL_FALLBACK = "gpt-5-search-api"

# ==============================
# INITIALIZATION
# ==============================
client = OpenAI()

print(f"📥 Loading dataset: {HF_DATASET}")
dataset = load_dataset(HF_DATASET, split="train")

df = dataset.to_pandas()
print(f"✅ Loaded {len(df)} rows")
print("Columns:", df.columns.tolist())

commented_x86 = []
start_idx = 0

# Uncomment to resume:
# existing = pd.read_parquet(OUTPUT_PATH)
# commented_x86 = list(existing["x86_commented"])
# start_idx = len(commented_x86)

print(f"Resuming from row {start_idx}")

# ==============================
# MAIN LOOP
# ==============================
for idx, row in df.iloc[start_idx:].iterrows():
    x86_code = row["x86"]

    prompt = f"""
You are an expert reverse-engineer and low-level assembly instructor.

Rewrite the following x86-64 assembly code **exactly as-is**, adding short inline comments using `;`.

Rules:
- Only meaningful semantic comments (control flow, pointers, calls, key arithmetic).
- DO NOT explain instruction syntax.
- DO NOT add or remove instructions, directives, or labels.
- DO NOT change indentation or formatting.
- No text outside code. Only inline comments.
- Keep comments under ~8 words.

x86 code:
{x86_code}
"""

    commented = None

    for model_name in [MODEL_PRIMARY, MODEL_FALLBACK]:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You annotate x86-64 assembly concisely without changing code structure."},
                    {"role": "user", "content": prompt}
                ],
            )

            commented = response.choices[0].message.content.strip()

            # Remove accidental code fences
            if commented.startswith("```"):
                commented = commented.split("```")[1].replace("x86", "").strip()

            print(f"[{idx+1}/{len(df)}] ✔ Commented {row['file']} using {model_name}")
            break

        except NotFoundError:
            print(f"[{idx+1}/{len(df)}] ⚠️ Model '{model_name}' not found, switching...")
            continue

        except Exception as e:
            print(f"[{idx+1}/{len(df)}] ❌ Error with {model_name}: {e}")
            commented = x86_code
            time.sleep(2)
            break

    commented_x86.append(commented)
    time.sleep(0.5)

    # --------- CHECKPOINT SAVE ----------
    if (idx + 1) % CHECKPOINT_INTERVAL == 0:
        df_checkpoint = df.copy()
        df_checkpoint.loc[:idx, "x86_commented"] = commented_x86
        df_checkpoint.to_parquet(OUTPUT_PATH, index=False)
        print(f"💾 Checkpoint saved at row {idx+1}")

# ==============================
# FINAL SAVE
# ==============================
df["x86_commented"] = commented_x86
df.to_parquet(OUTPUT_PATH, index=False)

print(f"🏁 Finished! Saved full commented dataset to {OUTPUT_PATH}")
