# Training GPTOSS

Fine-tune the **GPTOSS** (from OpenAI or Unsloth) model using LoRA with the Multilingual-Thinking dataset.  
This repo supports multiple datasets, including multi-turn datasets and datasets without the “thinking” keyword.  
Additionally, this repo supports Distributed Data Parallelism, enabling significantly faster training.  

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

Fix Target is a key function that ensures label tokens are correctly masked and handled for all supported datasets during training.

- Automatically adds the `<|target|>` token after final assistant message headers:
  - `<|start|>assistant<|channel|>final<|message|>`
  - `<|start|>assistant<|message|>`
- Ensures proper masking of labels in different dataset.

Test the functionality using the provided script:

```bash
python ./fix_target/fix_target.py
```

## Distributed Data Parallelism

**Distributed Data Parallelism(DDP)** enables significantly faster training by efficiently distributing data and computation across multiple GPUs.

The results of using different models and varying numbers of GPUs.

| Model               |   1 GPU   |   2 GPUs  |   4 GPUs  | 
|:--------------------|----------:|----------:|----------:|
| unsloth/gpt-oss-20b |  7775.18  |  4792.44  |  2530.38  | 
| openai/gpt-oss-20b  |  running  |  running  |  running  |

```bash
#SBATCH --gres=gpu:<# of gpus>       # Specify how many GPUs to use, e.g., 1, 2, 4, etc.
```

## Training

Use the [Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) dataset from HuggingFace, and convert it to the [Harmony format](https://github.com/openai/harmony).

Fine-tune GPTOSS using LoRA efficient and scalable training.  
Submit your training jobs via Slurm for distributed GPU execution.  

```bash
sbatch job_tf.slurm            # using train_tf.py
sbatch job_unsloth.slurm       # using train_unsloth.py
```

## Inference

Merge the fine-tuned weights with the base GPTOSS model, and use the resulting model for inference.

```bash
python chat_template.py
```

## References

1. [OpenAI Model](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) – Guide on fine-tuning GPTOSS using OpenAI tools.
2. [Unsloth Model](https://huggingface.co/unsloth/gpt-oss-20b-GGUF) – Details and specifications of the Unsloth GPTOSS model.
3. [OpenAI Harmony](https://github.com/openai/harmony) – A framework for converting datasets into a standardized format for training GPTOSS.
4. [Multilingual-Thinking Dataset](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) – A multilingual dataset designed for demonstration purposes, used to fine-tune GPT models.
5. [PEFT: LoRA Fine-Tuning](https://huggingface.co/docs/trl/sft_trainer) – Documentation for Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

---