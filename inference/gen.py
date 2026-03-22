import json
import random

# Load JSON file
with open("eval_armv8_O2_local.json", "r") as f:
    data = json.load(f)

# Assume predictions are in data["pred"]
preds = data["pred"]

# Sample 3 random predictions
samples = random.sample(preds, 3)

# Write to op2.txt
with open("op2.txt", "w") as f:
    for i, pred in enumerate(samples, 1):
        f.write(f"=== SAMPLE {i} ===\n")
        f.write(pred)
        f.write("\n\n")

print("Wrote 3 random predictions to op2.txt")

