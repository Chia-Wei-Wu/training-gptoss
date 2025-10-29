#!/usr/bin/env python
# coding: utf-8

def download_dataset():
    from datasets import load_dataset
    return load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

def transfer_harmony(datasets):

    from datasets import Dataset
    from openai_harmony import Role, Message, SystemContent, Conversation, ReasoningEffort, load_harmony_encoding, HarmonyEncodingName

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    features = []

    for sample in datasets:
        
        # reasoning_lang = sample.get("reasoning_language","")
        user_message = sample.get("user","")
        analysis_message = sample.get("analysis", "")
        final_message = sample.get("final","")
        
        input_messages = [
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
            Message.from_role_and_content(Role.USER, user_message),
            Message.from_role_and_content(Role.ASSISTANT, analysis_message).with_channel("analysis"),
        ]

        output_messages = [
            Message.from_role_and_content(Role.ASSISTANT, final_message).with_channel("final"),
        ]

        input_convo = Conversation.from_messages(input_messages)
        output_convo = Conversation.from_messages(output_messages)

        input_encoding = encoding.render_conversation_for_training(input_convo)
        output_encoding = encoding.render_conversation_for_training(output_convo)

        features.append({
            "input_ids": input_encoding,
            "label": output_encoding
        })

    conversations = Dataset.from_list(features)
    return conversations


def train_model(train_dataset, model_name, result_path):

    import os, gc, torch
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    quantization_config = Mxfp4Config(dequantize=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        
        # Configure PyTorch to reduce memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.6,expandable_segments:True"
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Loading {model_name} with memory optimization...")

        # Load model without allocating full weights on GPU immediately

        model_kwargs = dict(
            attn_implementation="eager",
            dtype=torch.float16,
            quantization_config=quantization_config,
            use_cache=False,
            device_map="auto",
        )
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # LoRA for fine-tuning
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            target_parameters=[
                "7.mlp.experts.gate_up_proj",
                "7.mlp.experts.down_proj",
                "15.mlp.experts.gate_up_proj",
                "15.mlp.experts.down_proj",
                "23.mlp.experts.gate_up_proj",
                "23.mlp.experts.down_proj",
            ],
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        # TrainingArguments
        training_args = SFTConfig(
            output_dir=result_path,
            learning_rate=2e-4,
            gradient_checkpointing=True,
            num_train_epochs=1,
            logging_steps=1,
            per_device_train_batch_size=1, 
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr_rate": 0.1},         
            save_strategy="steps",
            save_steps=500,
            fp16=True,  
            bf16=False,
            optim="adamw_torch",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
        )

        trainer.train()

        # Save LoRA adapter
        save_path = os.path.join("results/lora_adapter")
        os.makedirs(save_path, exist_ok=True)
        peft_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # Save Full fine-tune Model 
        # base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
        # merged_model = PeftModel.from_pretrained(base_model, save_path)
        # merged_model = merged_model.merge_and_unload()
        # os.makedirs("results/ft_model_full", exist_ok=True)
        # merged_model.save_pretrained("results/ft_model_full")
        # tokenizer.save_pretrained("results/ft_model_full")

    except Exception as e:
        print(f"Training failed: {e}")
        raise

def main():
    result_path = "results"
    model_name = "openai/gpt-oss-20b"
    datasets = download_dataset()
    train_dataset = transfer_harmony(datasets)
    train_model(train_dataset, model_name, result_path)

if __name__ == "__main__":
    main()