#!/bin/bash
set -e

DATASET="NadineMostafa/Angha_Stack_with_Mac_ARM"
MAXLEN=8192
OFFSET=1000
SHARDSIZE=150000
ARM_MAC_COL="arm_mac"

echo "Generating x86_to_armv8mac..."
python down_shard.py \
  --dataset-id $DATASET \
  --dataset-name x86_to_armv8mac \
  --src-col x86 \
  --tgt-col $ARM_MAC_COL \
  --max-context-length $MAXLEN \
  --context-length-offset $OFFSET \
  --shard-size $SHARDSIZE \
  --system-prompt "You are a helpful coding assistant specialized in converting from x86-64 to ARMv8 macOS (AArch64) assembly." \
  --user-prompt "Convert the following x86-64 assembly code to ARMv8 macOS (AArch64) assembly:\n\`\`\`asm\n{asm}\n\`\`\`"

echo "Generating armv8mac_to_x86..."
python down_shard.py \
  --dataset-id $DATASET \
  --dataset-name armv8mac_to_x86 \
  --src-col $ARM_MAC_COL \
  --tgt-col x86 \
  --max-context-length $MAXLEN \
  --context-length-offset $OFFSET \
  --shard-size $SHARDSIZE \
  --system-prompt "You are a helpful coding assistant specialized in converting from ARMv8 macOS (AArch64) to x86-64 assembly." \
  --user-prompt "Convert the following ARMv8 macOS (AArch64) assembly code to x86-64 assembly:\n\`\`\`asm\n{asm}\n\`\`\`"

echo "Generating armv8mac_to_riscv..."
python down_shard.py \
  --dataset-id $DATASET \
  --dataset-name armv8mac_to_riscv \
  --src-col $ARM_MAC_COL \
  --tgt-col riscv \
  --max-context-length $MAXLEN \
  --context-length-offset $OFFSET \
  --shard-size $SHARDSIZE \
  --system-prompt "You are a helpful coding assistant specialized in converting from ARMv8 macOS (AArch64) to RISC-V RV64 assembly." \
  --user-prompt "Convert the following ARMv8 macOS (AArch64) assembly code to RISC-V RV64 assembly:\n\`\`\`asm\n{asm}\n\`\`\`"

echo "Generating riscv_to_armv8mac..."
python down_shard.py \
  --dataset-id $DATASET \
  --dataset-name riscv_to_armv8mac \
  --src-col riscv \
  --tgt-col $ARM_MAC_COL \
  --max-context-length $MAXLEN \
  --context-length-offset $OFFSET \
  --shard-size $SHARDSIZE \
  --system-prompt "You are a helpful coding assistant specialized in converting from RISC-V RV64 to ARMv8 macOS (AArch64) assembly." \
  --user-prompt "Convert the following RISC-V RV64 assembly code to ARMv8 macOS (AArch64) assembly:\n\`\`\`asm\n{asm}\n\`\`\`"

echo "All MAC datasets generated successfully."