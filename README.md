# Training GPTOSS

Fine-tune the **GPTOSS** (from Openai or Unsloth) model using **LoRA** with the HuggingFaceH4/Multilingual-Thinking dataset.

## Environment Setup
Set up the training environment with conda.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh

cd training-gptoss
conda create -n env_training python=3.11.5 -y
conda activate env_training

pip install --upgrade pip

cd requirements
pip install -r tf.txt           # using train_tf.py
pip install -r unsloth.txt      # using train_unsloth.py
```

## Fix Target

**Fix Target** is a utility to properly mask labels in your dataset. This ensures that target tokens are correctly handled during training.

- Automatically adds the `<|target|>` token after relevant message headers:
  - `<|start|>assistant<|channel|>final<|message|>`
  - `<|start|>assistant<|message|>`
- Ensures proper masking of labels in your data.

Test the functionality using the provided script:

```bash
python ./fix_target/fix_target.py
```

## Distributed Data Parallelism

**Distributed Data Parallelism(DDP)** is the method that can speed up the training process by distributing data across multiple GPUs.

```bash
#SBATCH --gres=gpu:<# of gpus>            # located in job_unsloth.slurm or job_tf.slurm
```

## Training

Use the **[Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)** dataset, converted to **Harmony conversation format**.

Fine-tune GPTOSS using **LoRA** for efficient training.
Submit your training job with Slurm:

```bash
sbatch job_tf.slurm            # using train_tf.py
sbatch job_unsloth.slurm       # using train_unsloth.py
```

## Inference

Merge the fine-tuned weights with the base model, and use this model for inference.

```bash
python chat_template.py
```

## References

1. [OpenAI Harmony](https://github.com/openai/harmony)
2. [Multilingual-Thinking Dataset](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
3. [PEFT: LoRA Fine-Tuning](https://huggingface.co/docs/peft/index)
4. [Unsloth Model Card](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)

---
