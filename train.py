#!/usr/bin/env python
# coding: utf-8


def download_dataset():
    from datasets import load_dataset
    return load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

def transfer_harmony(datasets):
    from openai_harmony import Role, Message, SystemContent, Conversation
    conversations = []
    for sample in datasets:
        messages = sample.get("messages", [])
        conversation_messages = []
        conversation_messages.append(Message.from_role_and_content(Role.SYSTEM, SystemContent.new()))

        for msg in messages:
            role_str = msg.get("role", "").lower()
            content = msg.get("content", "")
            if not content.strip():
                continue

            if role_str in ["user", "human"]:
                conversation_messages.append(Message.from_role_and_content(Role.USER, content))
            elif role_str in ["assistant", "ai"]:
                conversation_messages.append(Message.from_role_and_content(Role.ASSISTANT, content))

        if conversation_messages:
            convo = Conversation.from_messages(conversation_messages)
            conversations.append(convo)

    return conversations

def build_dataset(conversations, model_name, max_length=2048):
    from datasets import Dataset
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Role
    from transformers import AutoTokenizer
    
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    features = []
    for convo in conversations:
        rendered = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
        if isinstance(rendered, list):
            rendered = "".join([str(r) for r in rendered])
        else:
            rendered = str(rendered)
        tokenized = tokenizer(rendered, return_tensors="pt")["input_ids"][0]

        # Sequence packing: split long sequences if exceeds max_length
        if len(tokenized) > max_length:
            for i in range(0, len(tokenized), max_length):
                features.append({
                    "input_ids": tokenized[i:i+max_length],
                    "labels": tokenized[i:i+max_length]
                })
        else:
            features.append({
                "input_ids": tokenized,
                "labels": tokenized
            })

    train_dataset = Dataset.from_list(features)
    return train_dataset

def train_model(train_dataset, model_name):
    import os, gc, torch
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
    try:
        
        # Configure PyTorch to reduce memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.6,expandable_segments:True"
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Loading {model_name} with memory optimization...")

        # Load model without allocating full weights on GPU immediately
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype=torch.float16, 
            device_map="auto"
        )
        model.gradient_checkpointing_enable()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # LoRA for fine-tuning
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        # TrainingArguments
        training_args = TrainingArguments(
            output_dir="results",  
            per_device_train_batch_size=1, 
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            logging_steps=50,
            save_strategy="steps",
            save_steps=500,
            fp16=True,  
            bf16=False,
            optim="adamw_torch",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
        )

        trainer.train()

        # Save LoRA adapter
        save_path = os.path.join("results/lora_adapter")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # Save Full fine-tune Model
        base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
        merged_model = PeftModel.from_pretrained(base_model, save_path)
        merged_model = merged_model.merge_and_unload()
        os.makedirs("results/ft_model_full", exist_ok=True)
        merged_model.save_pretrained("results/ft_model_full")
        tokenizer.save_pretrained("results/ft_model_full")

    except Exception as e:
        print(f"Training failed: {e}")
        raise

def main():
    model_name = "openai/gpt-oss-20b"
    datasets = download_dataset()
    conversations = transfer_harmony(datasets)
    train_dataset = build_dataset(conversations, model_name, max_length=2048)
    train_model(train_dataset, model_name)

if __name__ == "__main__":
    main()