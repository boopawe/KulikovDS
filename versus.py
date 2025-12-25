print("\n=== СРАВНЕНИЕ (BASE vs LORA) ===")
for prompt in test_prompts:
    print("\n" + "=" * 60)
    print(f"PROMPT: {prompt}")
    print("\nБАЗОВАЯ МОДЕЛЬ:")
    print(base_answers[prompt])
    print("\nLORA-МОДЕЛЬ:")
    print(lora_answers[prompt])
