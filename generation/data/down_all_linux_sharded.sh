#!/bin/bash
set -e

DATASET="NadineMostafa/Angha_Stack_all_ISAs_o2"
MAXLEN=8192
OFFSET=1000
SHARDSIZE=150000

gen () {
  local name="$1"
  local src="$2"
  local tgt="$3"
  local sys="$4"
  local user="$5"

  echo "Generating ${name}..."
  python down_shard.py \
    --dataset-id "$DATASET" \
    --dataset-name "$name" \
    --src-col "$src" \
    --tgt-col "$tgt" \
    --max-context-length "$MAXLEN" \
    --context-length-offset "$OFFSET" \
    --shard-size "$SHARDSIZE" \
    --system-prompt "$sys" \
    --user-prompt "$user"
}

gen x86_to_armv8 x86 arm \
  "You are a helpful coding assistant specialized in converting from x86-64 to ARMv8 (AArch64) assembly." \
  "Convert the following x86-64 assembly code to ARMv8 (AArch64) assembly:\n\`\`\`asm\n{asm}\n\`\`\`"

gen armv8_to_x86 arm x86 \
  "You are a helpful coding assistant specialized in converting from ARMv8 (AArch64) to x86-64 assembly." \
  "Convert the following ARMv8 (AArch64) assembly code to x86-64 assembly:\n\`\`\`asm\n{asm}\n\`\`\`"

gen x86_to_riscv x86 riscv \
  "You are a helpful coding assistant specialized in converting from x86-64 to RISC-V RV64 assembly." \
  "Convert the following x86-64 assembly code to RISC-V RV64 assembly:\n\`\`\`asm\n{asm}\n\`\`\`"

gen riscv_to_x86 riscv x86 \
  "You are a helpful coding assistant specialized in converting from RISC-V RV64 to x86-64 assembly." \
  "Convert the following RISC-V RV64 assembly code to x86-64 assembly:\n\`\`\`asm\n{asm}\n\`\`\`"

gen armv8_to_riscv arm riscv \
  "You are a helpful coding assistant specialized in converting from ARMv8 (AArch64) to RISC-V RV64 assembly." \
  "Convert the following ARMv8 (AArch64) assembly code to RISC-V RV64 assembly:\n\`\`\`asm\n{asm}\n\`\`\`"

gen riscv_to_armv8 riscv arm \
  "You are a helpful coding assistant specialized in converting from RISC-V RV64 to ARMv8 (AArch64) assembly." \
  "Convert the following RISC-V RV64 assembly code to ARMv8 (AArch64) assembly:\n\`\`\`asm\n{asm}\n\`\`\`"

echo "All Linux datasets generated successfully."