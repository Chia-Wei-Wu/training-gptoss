import torch, os, json, re
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only, standardize_sharegpt


def load_model(model_name, max_token):
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        dtype = None,
        max_seq_length = max_token,
        load_in_4bit = True,
        device_map="auto"
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

def setup_dataset():

    messages = [
        # Single-turn without system_prompt and without thinking
        {"messages": [
            {"role": "user", "content": "Please calculate 2 + 3"},
            {"role": "assistant", "content": "2 + 3 = 5"}]},

        # Single-turn without system_prompt and with thinking
        {"messages": [
            {"role": "user", "content": "Please calculate 2 + 3"},
            {"role": "assistant", "content": "Add 2 and 3 to get 5", "thinking": "First, sum the two numbers"}]},

        # Single-turn with system_prompt and without thinking
        {"messages": [
            {"role": "system", "content": "You are a calculation assistant"},
            {"role": "user", "content": "Please calculate 2 + 3"},
            {"role": "assistant", "content": "5"}]},

        # Single-turn with system_prompt and with thinking
        {"messages": [
            {"role": "system", "content": "You are a calculation assistant"},
            {"role": "user", "content": "Please calculate 2 + 3"},
            {"role": "assistant", "content": "Add 2 and 3 to get 5", "thinking": "Calculate the sum of the numbers"}]},

        # Multi-turn without system_prompt and without thinking
        {"messages": [
            {"role": "user", "content": "How's the weather today?"},
            {"role": "assistant", "content": "It is sunny today"},
            {"role": "user", "content": "And tomorrow?"},
            {"role": "assistant", "content": "It might rain tomorrow"}]},

        # Multi-turn without system_prompt and with thinking
        {"messages": [
            {"role": "user", "content": "How's the weather today?"},
            {"role": "assistant", "content": "Checking weather data... It is sunny today"},
            {"role": "user", "content": "And tomorrow?"},
            {"role": "assistant", "content": "Analyzing forecast... It might rain tomorrow", "thinking": "Analyze tomorrow's weather pattern"}]},

        # Multi-turn with system_prompt and without thinking
        {"messages": [
            {"role": "system", "content": "You are a weather assistant"},
            {"role": "user", "content": "How's the weather today?"},
            {"role": "assistant", "content": "It is sunny today"},
            {"role": "user", "content": "And tomorrow?"},
            {"role": "assistant", "content": "It might rain tomorrow"}]},

        # Multi-turn with system_prompt and with thinking
        {"messages": [
            {"role": "system", "content": "You are a weather assistant"},
            {"role": "user", "content": "How's the weather today?"},
            {"role": "assistant", "content": "Analyzing data... It is sunny today"},
            {"role": "user", "content": "And tomorrow?"},
            {"role": "assistant", "content": "Forecast shows rain tomorrow", "thinking": "Analyze tomorrow's weather forecast"}]}]

    return messages


def fix_target(text):

    text = re.sub(
        r"<\|start\|>assistant<\|message\|>",
        "<|start|>assistant<|channel|>final<|message|>",
        text)
    
    pattern = r"(<\|start\|>assistant<\|channel\|>final<\|message\|>)"
    matches = list(re.finditer(pattern, text))

    if not matches:
        print("The function did not run successfully.")
        breakpoint()
        return text 

    last_match = matches[-1]
    start, end = last_match.span()
    new_text = text[:end] + "<|target|>" + text[end:]

    return new_text


def formatting_prompts_func(batch, tokenizer):

    convos = batch["messages"]
    
    texts = []
    
    for convo in convos:
        for msg in convo:
            if msg.get("thinking") is None:
                msg.pop("thinking", None)
        formatted_text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        text = fix_target(formatted_text)
        texts.append(text)
    
    return {"text": texts}


def test_model(model, tokenizer, train_dataset, result_path):

    training_args = SFTConfig(
        output_dir = result_path,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
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
        dataset_num_proc=4,
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
        response_part="<|message|><|target|>")

    trainer = train_on_responses_only(trainer, **gpt_oss_kwargs, num_proc=4)

    # print result
    for i in range(0, 7):
        # breakpoint()
        text_input_ids = tokenizer.decode(trainer.train_dataset[i]["input_ids"])
        print(f"text_input_ids:\n{text_input_ids}")
        text_label = tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[i]["labels"]]).replace(tokenizer.pad_token, "")
        print(f"text_label:\n{text_label}")


def main():
    
    model_name = "unsloth/gpt-oss-20b"
    result_path = "results"
    max_token = 4096

    model, tokenizer = load_model(model_name, max_token)
    
    test_dataset = setup_dataset()
    test_dataset = Dataset.from_list(test_dataset)
    test_dataset = standardize_sharegpt(test_dataset)
    test_dataset = test_dataset.map(formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}, batched=True)

    test_model(model, tokenizer, test_dataset, result_path)   

    target_id = tokenizer.convert_tokens_to_ids("<|target|>")
    print(target_id)      #200019

if __name__ == "__main__":
    main()