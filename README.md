# 🧠 AI Tutor — Self-Improving Every Night
### Powered by Groq + Autoresearch (inspired by karpathy/autoresearch)

## Setup
```bash
pip install streamlit openai
export GROQ_API_KEY=your_key_here
streamlit run app.py
```

## File structure
- `app.py`            — Streamlit UI (student + teacher views)
- `prompts.py`        — THE FILE THE AGENT EDITS (like train.py in autoresearch)
- `autoresearch.py`   — The overnight loop (mutate → eval → keep/discard)
- `tutor.py`          — Core AI functions (terminal fallback)

## Run autoresearch overnight
```bash
python autoresearch.py 20   # 20 experiments
python autoresearch.py log  # view log
```
