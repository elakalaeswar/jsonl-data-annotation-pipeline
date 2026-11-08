# Automated Ground-Truth Annotation & JSONL Data Pipeline

A Python data engineering script that ingests raw conversational text, cleans whitespace/nulls, eliminates duplicates, and formats data into standard JSONL chat schema for Supervised Fine-Tuning (SFT).

## Features
- **Deduplication & Hygiene:** Strips trailing whitespace, drops null/empty records, and eliminates duplicate prompts.
- **Token Estimation & Metadata:** Annotates each record with character-based heuristic token counts.
- **SFT Schema Formatting:** Exports clean dialogue pairs into standard user/assistant chat format for LLM training pipelines.

## Usage
```bash
pip install -r requirements.txt
python pipeline.py
