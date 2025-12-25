from peft import PeftModel

lora_model = PeftModel.from_pretrained(
    base_model,
    "./my-lora-model-best"
)

print("\n=== ОБУЧЕННАЯ МОДЕЛЬ (ПОСЛЕ LoRA) ===")
lora_answers = {}
for prompt in test_prompts:
    answer = generate_plain(lora_model, tokenizer, prompt)
    lora_answers[prompt] = answer
    print(f"\nPrompt: {prompt}\nОтвет: {answer}")
