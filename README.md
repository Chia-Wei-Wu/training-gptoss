# Training GPTOSS

Fine-tune the **GPTOSS** model using **LoRA** with the HuggingFaceH4/Multilingual-Thinking dataset.

## Environment Setup

Set up a Python virtual environment and install the required packages:

```bash
cd training-gptoss
python -m venv env_training
source ./env_training/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training

We use the **[Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)** dataset, converted to **Harmony conversation format**.

Fine-tune GPTOSS using **LoRA** with GPU memory optimizations for efficient training.

Run training:

```bash
sbatch job.slurm
```

## References

1. [OpenAI Harmony](https://github.com/openai/harmony) – Harmony conversation format.
2. [Multilingual-Thinking Dataset](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) – Dataset Info.
3. [PEFT: LoRA Fine-Tuning](https://huggingface.co/docs/peft/index) – LoRA Info

---
