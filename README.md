# Training GPTOSS

Fine-tune the **GPTOSS** model using **LoRA** with the HuggingFaceH4/Multilingual-Thinking dataset.

## Environment Setup
You can set up the training environment with conda.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh

cd training-gptoss
conda create -n env_training python=3.11.5 -y
conda activate env_training

pip install --upgrade pip
pip install -r requirements.txt
```

## Training

We use the **[Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)** dataset, converted to **Harmony conversation format**.

Fine-tune GPTOSS using **LoRA** with GPU memory optimizations for efficient training.
Submit your training job with Slurm:

```bash
sbatch job-conda.slurm
```

## References

1. [OpenAI Harmony](https://github.com/openai/harmony)
2. [Multilingual-Thinking Dataset](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
3. [PEFT: LoRA Fine-Tuning](https://huggingface.co/docs/peft/index)

---
