import os, torch, re, json
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def pre_model(model_name, max_token):

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"  

    quantization_config = Mxfp4Config(dequantize=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = dict(
        attn_implementation="eager",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        # device_map="auto",
        device_map={'':device}
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer.add_special_tokens({
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",})

    model.resize_token_embeddings(len(tokenizer))
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def find_labels(text, tokenizer):
    
    pattern = r"(<\|start\|>assistant<\|channel\|>final)?<\|message\|>"
    matches = list(re.finditer(pattern, text))

    if not matches:
        print("The function did not run successfully.")
        return None 

    last_match = matches[-1]
    start, end = last_match.span()

    input_ids = tokenizer(text).input_ids
    labels_text = text[end:]
    label_ids = tokenizer(labels_text).input_ids

    num_ignore = len(input_ids) - len(label_ids)
    labels = [-100] * num_ignore + label_ids

    return labels


def formatting_prompts_func(batch, tokenizer):
    
    convos = batch["messages"]
    
    input_ids_list = []
    labels_list = []

    for convo in convos:
        for msg in convo:
            if msg.get("thinking") is None:
                msg.pop("thinking", None)

        formatted_text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        labels_text = find_labels(formatted_text, tokenizer)

        input_ids = tokenizer(formatted_text, truncation=True)["input_ids"]

        input_ids_list.append(input_ids)
        labels_list.append(labels_text)

    return {"input_ids": input_ids_list, "labels": labels_list}


def train(model, tokenizer, train_dataset, result_path, max_token):
    
    training_args = SFTConfig(
        output_dir = result_path,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        # num_train_epochs = 5,
        max_steps = 3,
        learning_rate = 2e-4,
        logging_steps = 1,
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        dataset_num_proc=4,
        max_length = max_token,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = None,
    )
    
    ### print result
    # text_input_ids = tokenizer.decode(trainer.train_dataset[100]["input_ids"])
    # print(f"text_input_ids:\n{text_input_ids}")
    # text_label = tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")
    # print(f"text_label:\n{text_label}")
    
    trainer.train()

    ### Save LoRA adapter
    save_path = os.path.join(f"{result_path}/lora_adapter")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, save_embedding_layers=True)
    tokenizer.save_pretrained(save_path)


def main():

    model_name = "openai/gpt-oss-20b"
    dataset_path = "HuggingFaceH4/Multilingual-Thinking"
    result_path = "results"
    max_token = 4096

    model, tokenizer = pre_model(model_name, max_token)

    train_dataset = load_dataset(dataset_path, split="train")

    train_dataset = Dataset.from_list(train_dataset)
    train_dataset = train_dataset.map(formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}, batched=True)

    print("Happy Training GPTOSS...", flush=True)

    train(model, tokenizer, train_dataset, result_path, max_token)

if __name__ == "__main__":
    main()