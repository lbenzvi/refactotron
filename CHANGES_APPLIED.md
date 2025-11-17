# Training Configuration Changes - Version 2

## Summary

This document describes the configuration fixes applied to create `refactotron_training_v2.ipynb` from the previous training notebooks. The changes address critical issues that were limiting model performance and training stability.

## Changes Applied

### 1. Data Collator Correction
**Previous:** `DataCollatorForSeq2Seq`
**Current:** `DataCollatorForLanguageModeling`

Changed from seq2seq collator (designed for encoder-decoder models like T5/BART) to language modeling collator appropriate for causal language models like StarCoder.

### 2. GPU Precision Detection
**Previous:** Fixed `bf16=True` setting
**Current:** Dynamic precision based on GPU capabilities

Added automatic detection to use FP16 on T4 GPUs and BF16 on A100/H100 GPUs, since T4 does not support BF16 operations.

### 3. Label Smoothing Removal
**Previous:** `label_smoothing_factor=0.05`
**Current:** Removed

Eliminated label smoothing as it is incompatible with masked labels (-100 padding) used in causal language modeling.

### 4. Evaluation Configuration Optimization
**Previous:** `eval_steps=500`
**Current:** `eval_steps=1000`, added `eval_accumulation_steps=4` and `fp16_full_eval=False`

Adjusted evaluation to occur after warmup period completes, added gradient accumulation during evaluation for memory efficiency, and disabled FP16 for evaluation to prevent numerical instability.

## Expected Impact

**Previous validation loss:** 0.637
**Target validation loss:** 0.48-0.55
**Expected improvement:** Approximately 30% reduction in validation loss

## Files

- **Training notebook:** `refactotron_training_v2.ipynb`
- **Data files:** `train_enhanced.jsonl`, `validation_enhanced.jsonl` (unchanged)
- **Previous versions:** Retained for reference

## Next Steps

Upload `refactotron_training_v2.ipynb` to Google Colab and train with the corrected configuration to achieve improved performance.
