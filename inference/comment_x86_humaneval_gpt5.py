import pandas as pd
import time
from openai import OpenAI, NotFoundError

# ==============================
# CONFIGURATION
# ==============================
INPUT_PATH = "HumanEval-armv8-O2-clang-native.parquet"
OUTPUT_PATH = "HumanEval-armv8-O2-clang-native-commented.parquet"
CHECKPOINT_INTERVAL = 5
MODEL_PRIMARY = "gpt-5"
MODEL_FALLBACK = "gpt-5-search-api"

# ==============================
# INITIALIZATION
# ==============================
client = OpenAI()

df = pd.read_parquet(INPUT_PATH)
print(f"✅ Loaded {len(df)} rows from {INPUT_PATH}")
print("Columns:", df.columns.tolist())

commented_x86 = []
start_idx = 0

# If resuming from a partial run:
#df_existing = pd.read_parquet(OUTPUT_PATH)
#commented_x86 = list(df_existing["x86_content"])
#start_idx = len(commented_x86)
print(f"Resuming from checkpoint at row {start_idx}")

# ==============================
# MAIN LOOP
# ==============================
for idx, row in df.iloc[start_idx:].iterrows():
    x86_code = row["x86_content"]

    prompt = f"""
You are an expert reverse-engineer and low-level assembly instructor.

Rewrite the following x86-64 assembly code **exactly as-is**, but add **helpful inline comments** at the end of lines using `;`.

Commenting rules (important):
- Comment only what is meaningful to program semantics (control flow, pointer dereferencing, function calls, data movement).
- **Do NOT explain instruction syntax** (e.g., "add adds two values" or "mov moves data").
- Keep comments short (≤ 8 words), factual, and high-signal.
- Do **not** add or remove instructions, labels, or directives.
- Do **not** alter spacing or indentation.
- No explanations before or after the code. Only inline comments.

Example style:
    mov rax, rdi        ; load first argument
    call strlen         ; get length of string

x86-64 input code:
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

            # Clean up accidental code fences
            if commented.startswith("```"):
                commented = commented.split("```")[1].replace("x86", "").strip()

            print(f"[{idx+1}/{len(df)}] ✅ Commented {row['file_name']} using {model_name}")
            break

        except NotFoundError:
            print(f"[{idx+1}/{len(df)}] ⚠️ Model '{model_name}' not found, trying fallback...")
            continue
        except Exception as e:
            print(f"[{idx+1}/{len(df)}] ❌ Error with {model_name}: {e}")
            commented = x86_code
            time.sleep(2)
            break

    commented_x86.append(commented)
    time.sleep(0.5)

    if (idx + 1) % CHECKPOINT_INTERVAL == 0:
        df_checkpoint = df.copy()
        df_checkpoint.loc[:idx, "x86_content"] = commented_x86
        df_checkpoint.to_parquet(OUTPUT_PATH, index=False)
        print(f"💾 Checkpoint saved at row {idx+1} → {OUTPUT_PATH}")

# ==============================
# FINAL SAVE
# ==============================
df["x86_content"] = commented_x86
df.to_parquet(OUTPUT_PATH, index=False)
print(f"🏁 Finished! Saved full commented dataset to {OUTPUT_PATH}")

