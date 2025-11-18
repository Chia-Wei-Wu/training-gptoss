def download_base_model(model_name):
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config

    quantization_config = Mxfp4Config(dequantize=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = dict(
        attn_implementation="eager",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto"
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return model, tokenizer

def download_ft_model(model_name, lora_url):
    
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config

    quantization_config = Mxfp4Config(dequantize=True)
    tokenizer = AutoTokenizer.from_pretrained(lora_url)

    model_kwargs = dict(
        attn_implementation="eager",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto"
    )

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    model = PeftModel.from_pretrained(base_model, lora_url)
    model = model.merge_and_unload()

    return model, tokenizer

def chatting(model, tokenizer):
     
    reasoning_language = "German"
    system_prompt = f"reasoning language: {reasoning_language}"
    user_prompt = "¿Cuál es el capital de Australia?"  # Spanish for "What is the capital of Australia?"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": 512, 
        "do_sample": True, 
        "temperature": 0.6, 
        "top_p": None, 
        "top_k": None
    }

    output_ids = model.generate(input_ids, **gen_kwargs)
    response = tokenizer.batch_decode(output_ids)[0]
    response = response.split("<|channel|>final<|message|>")[-1].strip()
    response = response.replace("<|return|>", "").strip()

    return response


def main():

    # model_name = "openai/gpt-oss-20b"
    model_name = "unsloth/gpt-oss-20b"                             # unsloth model
    lora_url = "results-ver1/lora_adapter"

    # model, tokenizer = download_base_model(model_name)             # base model
    model, tokenizer = download_ft_model(model_name, lora_url)       # ft model  

    result = chatting(model, tokenizer)
    print(result)

if __name__ == "__main__":
    main()