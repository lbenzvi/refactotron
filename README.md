# Refactotron

A fine-tuned code refactoring model built on StarCoder-1B using LoRA (Low-Rank Adaptation) for efficient training. Refactotron improves Python code quality through intelligent refactoring suggestions.

## 🎯 Results

**Final Model Performance:**
- **Validation Loss**: 0.49
- **BLEU Score**: 72
- **Semantic Similarity**: 0.90

These metrics demonstrate strong performance in generating semantically correct and syntactically accurate refactored code.

## 🏗️ Architecture

- **Base Model**: `bigcode/starcoderbase-1b`
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: `c_proj`, `c_attn`
  - Dropout: 0.08

## 📊 Training Details

### Dataset
- **Training samples**: 39,812
- **Validation samples**: 4,976
- **Data format**: Enhanced JSONL with input/output pairs
- **Max sequence length**: 1024 tokens

### Training Configuration
```python
- Learning rate: 2e-5
- Scheduler: Cosine with warmup
- Warmup steps: 500
- Batch size: 1 (per device)
- Gradient accumulation: 8 steps (effective batch size: 8)
- Epochs: 5
- Weight decay: 0.02
- Precision: FP16 (training), FP32 (evaluation)
- Optimizer: AdamW (fused)
- Gradient clipping: 1.0
```

### Hardware
- **GPU**: NVIDIA Tesla T4
- **Training time**: ~12-15 hours
- **Memory**: ~12-14 GB VRAM

## 🔧 Critical Implementation Fixes

During development, several critical issues were identified and resolved:

### 1. **Data Collator Issue** ✅
**Problem**: Used `DataCollatorForSeq2Seq` for a causal language model
**Solution**: Switched to `DataCollatorForLanguageModeling(mlm=False)`
- StarCoder is a causal LM, not an encoder-decoder model
- Wrong collator caused NaN validation loss

### 2. **Label Smoothing with Masked Labels** ✅
**Problem**: `label_smoothing_factor=0.05` caused NaN with `-100` masked labels
**Solution**: Removed label smoothing entirely
- Label smoothing redistributes probability across all tokens
- Incompatible with masked labels (`-100` for input tokens)

### 3. **Mixed Precision Instability** ✅
**Problem**: BF16 precision caused numerical instability on T4 GPU
**Solution**: Used FP16 for training, FP32 for evaluation
- Added `fp16_full_eval=False` to prevent NaN during validation
- Added `eval_accumulation_steps=4` for stability with batch size 1

### 4. **Evaluation Timing** ✅
**Problem**: Early evaluation before warmup could cause instability
**Solution**: Set `eval_steps=1000` to evaluate after warmup completes
- Ensures model is properly initialized before first validation

### 5. **Label Masking Implementation** ✅
**Problem**: Loss computed on both input and output tokens
**Solution**: Proper label masking in tokenization
```python
# Input tokens: -100 (ignored in loss)
# Output tokens: actual token IDs (used in loss)
labels = [-100] * len(inp_ids) + out_ids
```

## 📁 Project Structure

```
refactotron/
├── refactotron_training_FINAL_OPTIMIZED_FIXED.ipynb  # Final working training notebook
├── create_training_data_fixed.py                      # Data preparation script
├── enhanced_data_generation.ipynb                     # Enhanced dataset generation
├── data/                                              # Training data directory
│   ├── train_enhanced.jsonl                          # Training dataset (~60 MB)
│   └── validation_enhanced.jsonl                     # Validation dataset (~7.5 MB)
└── src/                                              # Source code
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install transformers datasets peft accelerate bitsandbytes
```

### 2. Prepare Data
Ensure you have the enhanced training files:
- `train_enhanced.jsonl` (~60 MB, 39,812 samples)
- `validation_enhanced.jsonl` (~7.5 MB, 4,976 samples)

### 3. Train Model
Use the `refactotron_training_FINAL_OPTIMIZED_FIXED.ipynb` notebook:
1. Mount Google Drive
2. Check GPU availability (requires T4 or better)
3. Upload training data
4. Authenticate with HuggingFace
5. Run training cells (12-15 hours)

### 4. Resume from Checkpoint
If training is interrupted:
```python
trainer.train(resume_from_checkpoint=True)
```

## 📈 Expected Training Progression

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 1000 | ~0.69        | ~0.68          |
| 2000 | ~0.67        | ~0.66          |
| 3000 | ~0.68        | ~0.66          |
| Final| ~0.48-0.50   | ~0.48-0.50     |

## 💡 Usage

### Load Fine-tuned Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "bigcode/starcoderbase-1b",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "/path/to/refactotron_lora_FINAL"
)

tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
```

### Generate Refactored Code
```python
input_text = """### Refactor the following Python code to improve quality:

def calculate(x, y):
    result = x + y
    return result
"""

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=512, temperature=0.2)
refactored_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(refactored_code)
```

## 🐛 Troubleshooting

### NaN Validation Loss
- ✅ Use `DataCollatorForLanguageModeling` (not `DataCollatorForSeq2Seq`)
- ✅ Remove `label_smoothing_factor`
- ✅ Set `fp16_full_eval=False`
- ✅ Use `eval_accumulation_steps=4`

### High Training Loss (>1.0)
- Check data format (should be enhanced JSONL files)
- Verify label masking is working correctly
- Ensure correct data collator is used

### Training Stops After Closing Laptop
- Keep browser tab open (just minimize, don't close)
- Resume from checkpoint: `trainer.train(resume_from_checkpoint=True)`
- Consider using tmux/screen for long-running sessions

### Out of Memory
- Reduce `per_device_train_batch_size` (already at 1)
- Enable gradient checkpointing (already enabled)
- Use smaller sequence length if possible

## 📝 Key Learnings

1. **Data collator must match model architecture** - Using seq2seq collators with causal LMs causes evaluation issues
2. **Label smoothing incompatible with masked labels** - Remove when using `-100` for masking
3. **Precision matters for numerical stability** - FP16 works better than BF16 on T4 GPUs
4. **Proper label masking is critical** - Only compute loss on output tokens, not input
5. **Evaluation timing matters** - Wait until after warmup for stable metrics

## 🎓 Academic Context

This project was developed for DS340 at Boston University by Liam Ben-Zvi and Aryan Lunkad.

### Key Contributions:
1. Novel benchmark-optimized data collection strategy with 39,812 enhanced training samples
2. Comprehensive debugging and fixing of critical training issues (data collator, label smoothing, precision handling)
3. Systematic analysis of LoRA fine-tuning achieving strong performance (0.49 validation loss, 72 BLEU, 0.90 similarity)
4. Complete documentation of training methodology and troubleshooting approaches

## 📄 Citation
```bibtex
@misc{refactotron2024,
  author = {Ben-Zvi, Liam and Lunkad, Aryan},
  title = {Refactotron: Automated Code Refactoring with Fine-tuned Language Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/lbenzvi/refactotron}
}
```

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Base Model**: [StarCoder](https://huggingface.co/bigcode/starcoderbase-1b) by BigCode
- **LoRA Implementation**: [PEFT](https://github.com/huggingface/peft) by HuggingFace
- **Training Framework**: [Transformers](https://github.com/huggingface/transformers) by HuggingFace
- **Boston University SCC** for computational resources

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Training Status**: ✅ Successfully trained and validated
**Last Updated**: November 11, 2025
