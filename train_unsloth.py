def pre_model(model_name, max_token):

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        dtype = None,
        max_seq_length = max_token,
        load_in_4bit = True,
        full_finetuning = False,
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

def formatting_prompts_func(data, tokenizer):

    messages = data["messages"]

    conversations = [
        tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=False
        ) for msg in messages
    ]
    return {"text": conversations}

def train(model, tokenizer, train_dataset, result_path):

    import os
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, 
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = result_path,
        report_to = "none",
    )
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        args=training_args,
        eval_dataset=None,
    )

    trainer.train()

    # Save LoRA adapter
    save_path = os.path.join(f"{result_path}/unsloth_lora_adapter")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def main():
    
    from datasets import load_dataset
    from unsloth.chat_templates import standardize_sharegpt

    model_name = "unsloth/gpt-oss-20b"
    dataset_path = "HuggingFaceH4/Multilingual-Thinking"
    result_path = "results"
    max_token = 4096

    model, tokenizer = pre_model(model_name, max_token)
    
    train_dataset = load_dataset(dataset_path, split="train")
    train_dataset = standardize_sharegpt(train_dataset)
    train_dataset = train_dataset.map(lambda x: formatting_prompts_func(x, tokenizer=tokenizer), batched=True)
    # print(train_dataset[0]['text'])

    train(model, tokenizer, train_dataset, result_path)


if __name__ == "__main__":
    main()