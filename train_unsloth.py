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
        # device_map="auto"
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

    tokenizer.add_special_tokens({"additional_special_tokens": ["<|target|>"]})

    return model, tokenizer


def fix_target(text):
    
    pattern = r"(<\|start\|>assistant(?:<\|channel\|>final)?<\|message\|>)"
    matches = list(re.finditer(pattern, text))

    if not matches:
        print("The function did not run successfully.")
        breakpoint()
        return text 

    last_match = matches[-1]
    start, end = last_match.span()
    new_text = text[:end] + "<|target|>" + text[end:]

    return new_text


def formatting_prompts_func(example, tokenizer):

    convos = example["messages"]
    
    texts = []
    
    for convo in convos:
        for msg in convo:
            if msg.get("thinking") is None:
                msg.pop("thinking", None)
        formatted_text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        text = fix_target(formatted_text)
        texts.append(text)
    
    return {"text": texts}

def train(model, tokenizer, train_dataset, result_path):

    training_args = SFTConfig(
        output_dir = result_path,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        num_train_epochs = 5,
        # max_steps = 10,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        dataset_num_proc=4,
    )
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        args = training_args,
        eval_dataset = None,
    )

    gpt_oss_kwargs = dict(
        instruction_part = "<|start|>user<|message|>", 
        response_part="<|message|><|target|>")

    trainer = train_on_responses_only(trainer, **gpt_oss_kwargs, num_proc=2)

    # text_input_ids = tokenizer.decode(trainer.train_dataset[100]["input_ids"])
    # print(f"text_input_ids:\n{text_input_ids}")
    # text_label = tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, "")
    # print(f"text_label:\n{text_label}")

    trainer.train()

    ### Save LoRA adapter
    save_path = os.path.join(f"{result_path}/lora_adapter")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def main():
    
    model_name = "unsloth/gpt-oss-20b"
    dataset_path = "HuggingFaceH4/Multilingual-Thinking"
    result_path = "results-ver3"
    max_token = 4096

    model, tokenizer = pre_model(model_name, max_token)
    
    train_dataset = load_dataset(dataset_path, split="train")

    train_dataset = standardize_sharegpt(train_dataset)
    train_dataset = train_dataset.map(formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}, batched=True)

    print("Happy Training GPTOSS...", flush=True)

    train(model, tokenizer, train_dataset, result_path)


if __name__ == "__main__":
    main()