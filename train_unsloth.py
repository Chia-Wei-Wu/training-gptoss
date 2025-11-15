#!/usr/bin/env python
# coding: utf-8

import torch, os, json, re
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only, standardize_sharegpt

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def pre_model(model_name, max_token):

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        dtype = None,
        max_seq_length = max_token,
        load_in_4bit = True,
        device_map={'':device}
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias = "none",   
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None, 
    )

    return model, tokenizer

def download_dataset():
    return load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

def fix_final_channel(text):

    pattern = r'(<\|start\|>assistant(?:<\|channel\|>[^<]*)?<\|message\|>)'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return text 

    for m in matches[:-1]:
        start, end = m.span()
        segment = text[start:end]
        cleaned = re.sub(r'<return>', '<end>', segment)
        text = text[:start] + cleaned + text[end:]
    
    matches = list(re.finditer(pattern, text))
    last_match = matches[-1]
    start, end = last_match.span()
    segment = text[start:end]

    if '<|channel|>final' not in segment:
        if '<|channel|' in segment:
            segment = re.sub(r'(<\|channel\|>[^<]*)', r'\1<|channel|>final', segment, count=1)
        else:
            segment = segment.replace('<|start|>assistant', '<|start|>assistant<|channel|>final', 1)
        text = text[:start] + segment + text[end:]
    
    return text

def formatting_prompts_func(batch, tokenizer):

    convos = batch["messages"]
    
    texts = []
    
    for convo in convos:
        for msg in convo:
            if msg.get("thinking") is None:
                msg.pop("thinking", None)
        formatted_text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        text = fix_final_channel(formatted_text)
        texts.append(text)
    
    return {"text": texts}

def train(model, tokenizer, train_dataset, result_path):

    training_args = SFTConfig(
        output_dir = result_path,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        # num_train_epochs = 1,
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        fp16=False, 
        bf16=False,
        save_strategy="steps",
        warmup_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
    )
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        args=training_args,
        eval_dataset=None,
    )

    gpt_oss_kwargs = dict(
        instruction_part = "<|start|>user<|message|>", 
        response_part="<|start|>assistant<|channel|>final<|message|>")

    ## check input_ids and labels for training_data 
    text_input_ids = tokenizer.decode(trainer.train_dataset[2]["input_ids"])
    print(f"text_input_ids:\n{text_input_ids}")
    text_label = tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[2]["labels"]]).replace(tokenizer.pad_token, "")
    print(f"text_label:\n{text_label}")


    # trainer.train()

    # trainer = train_on_responses_only(trainer, **gpt_oss_kwargs)

    ### Save LoRA adapter
    # save_path = os.path.join(f"{result_path}/unsloth_lora_adapter")
    # os.makedirs(save_path, exist_ok=True)
    # model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)


def main():
    
    model_name = "unsloth/gpt-oss-20b"
    dataset_path = "HuggingFaceH4/Multilingual-Thinking"
    result_path = "results"
    max_token = 4096

    model, tokenizer = pre_model(model_name, max_token)
    
    train_dataset = load_dataset(dataset_path, split="train")
    
    train_dataset = train_dataset[:3]

    train_dataset = standardize_sharegpt(train_dataset)
    train_dataset = train_dataset.map(formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}, batched=True)

    print("Happy Training GPTOSS...", flush=True)
    train(model, tokenizer, train_dataset, result_path)


if __name__ == "__main__":
    main()