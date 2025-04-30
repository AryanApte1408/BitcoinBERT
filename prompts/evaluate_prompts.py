import pandas as pd
from transformers import pipeline

def evaluate_on_prompts(model, tokenizer):
    prompts = [
        "Is it a good time to BUY Bitcoin?",
        "Should I SELL my Bitcoin holdings today?",
        "Would you BUY or SELL Bitcoin this week?",
        "Should I invest in Bitcoin right now?",
        "Is it wise to HOLD Bitcoin for the next month?"
    ]

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

    print("\nPrompt-Level Evaluation:")
    for p in prompts:
        res = pipe(p, truncation=True)[0]
        print(f"Prompt: {p}\n → Prediction: {res['label']} (confidence: {res['score']:.2f})\n")
