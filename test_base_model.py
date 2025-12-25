from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

test_prompts = [
    "Видеомонтаж — это",
    "Основная задача видеомонтажа заключается в том, что",
    "Хороший видеомонтаж отличается от любительского тем, что",
    "Для видеомонтажа чаще всего используют программы",
]

def generate_plain(model, tokenizer, prompt):
    text = f"Вопрос: {prompt}\nОтвет:"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Ответ:")[-1].strip()

print("=== БАЗОВАЯ МОДЕЛЬ (ДО ОБУЧЕНИЯ) ===")
base_answers = {}
for prompt in test_prompts:
    answer = generate_plain(base_model, tokenizer, prompt)
    base_answers[prompt] = answer
    print(f"\nPrompt: {prompt}\nОтвет: {answer}")
