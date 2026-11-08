import json
import os
import re
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_dataset(input_csv, output_jsonl):
    print("\n--- RUNNING JSONL DATA PIPELINE ---")
    df = pd.read_csv(input_csv)
    total_raw = len(df)

    df["prompt"] = df["prompt"].apply(clean_text)
    df["response"] = df["response"].apply(clean_text)
    df = df[(df["prompt"] != "") & (df["response"] != "")].drop_duplicates(subset=["prompt"])

    records = []
    for idx, row in df.iterrows():
        p_tokens = max(1, len(row["prompt"]) // 4)
        r_tokens = max(1, len(row["response"]) // 4)
        records.append({
            "id": f"SFT-{idx+1:05d}",
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]}
            ],
            "metadata": {
                "category": row.get("category", "General"),
                "total_tokens": p_tokens + r_tokens
            }
        })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processed: {total_raw} | Clean exported: {len(records)}\n")

if __name__ == "__main__":
    sample = {
        "prompt": [
            "Explain Newton's third law with an engineering example.",
            "Explain Newton's third law with an engineering example.",
            "What is the function of a boiler economizer?",
            ""
        ],
        "response": [
            "For every action, there is an equal and opposite reaction. In rocketry, downward thrust accelerates the vehicle upward.",
            "Duplicate entry.",
            "An economizer recovers waste heat from flue gases to preheat boiler feed water.",
            "Empty prompt."
        ],
        "category": ["Physics", "Physics", "Thermodynamics", "Invalid"]
    }
    pd.DataFrame(sample).to_csv("raw_conversations.csv", index=False)
    process_dataset("raw_conversations.csv", "clean_sft_dataset.jsonl")