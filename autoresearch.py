"""
autoresearch.py — Autonomous overnight research loop
=====================================================
Mirrors karpathy/autoresearch architecture exactly:

  train.py       →  prompts.py        (the file the agent edits)
  val_bpb        →  avg_score         (metric, higher is better)
  5-min training →  synthetic eval    (fast loop, no GPU needed)
  agent loop     →  this file         (mutate → evaluate → keep/discard)

Run this overnight:
  python autoresearch.py

It will run N experiments, each time:
  1. Read current prompts.py
  2. Ask Groq to mutate it (improve question/eval/explanation prompts)
  3. Evaluate the new prompts against student failure data
  4. If score improved → keep the new prompts.py
  5. If score dropped  → discard, restore previous
  6. Log everything to autoresearch_log.json
"""

import os
import json
import time
import shutil
import datetime
import importlib
import importlib.util
from pathlib import Path
import openai

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

DATA_FILE     = "student_data.json"
PROMPTS_FILE  = "prompts.py"
BACKUP_FILE   = "prompts_backup.py"
LOG_FILE      = "autoresearch_log.json"

N_EXPERIMENTS = 10   # how many mutations to try per overnight run
EVAL_SAMPLES  = 5    # synthetic test cases per experiment

# ── helpers ───────────────────────────────────────────────────────────────────

def chat(messages, temperature=0.7):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # ✅ نیا model
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def load_log():
    if Path(LOG_FILE).exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return {"experiments": [], "best_score": 0.0, "generation": 1}

def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def load_student_data():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"students": {}, "overnight_insights": []}

def load_prompts_module(filepath=PROMPTS_FILE):
    spec = importlib.util.spec_from_file_location("prompts", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read_prompts_source():
    with open(PROMPTS_FILE) as f:
        return f.read()

def write_prompts_source(source):
    with open(PROMPTS_FILE, "w") as f:
        f.write(source)

def backup_prompts():
    shutil.copy(PROMPTS_FILE, BACKUP_FILE)

def restore_prompts():
    shutil.copy(BACKUP_FILE, PROMPTS_FILE)

# ── evaluation (the "5-minute training run") ──────────────────────────────────

def evaluate_prompts(prompts_mod, student_data, generation):
    """
    Evaluates current prompts against real student failure data.
    Returns avg_score 0-10 (higher = better teaching quality).
    
    Strategy:
    - Pull real failed questions from student_data
    - Re-run them through the new prompts
    - Ask Groq to rate how good the explanation/feedback is
    """
    failures = []
    for student, sdata in student_data.get("students", {}).items():
        for attempt in sdata.get("attempts", []):
            if not attempt["correct"]:
                failures.append(attempt)

    # if no real data yet, use synthetic test cases
    if len(failures) < 3:
        failures = _synthetic_failures()

    # sample up to EVAL_SAMPLES
    import random
    sample = random.sample(failures, min(EVAL_SAMPLES, len(failures)))

    scores = []
    for attempt in sample:
        topic   = attempt.get("topic", "math")
        question = attempt.get("question", "What is 5 x 6?")
        wrong_answer = attempt.get("student_answer", "I don't know")

        # build eval prompt using the CURRENT prompts.py template
        eval_filled = prompts_mod.EVAL_PROMPT.format(
            question=question,
            correct_answer="(correct answer not stored)",
            student_answer=wrong_answer,
        )

        # ask a meta-evaluator: "how good is this feedback for a student?"
        meta_prompt = f"""You are evaluating the quality of an AI tutor's feedback system.

A student got this question wrong:
Question: {question}
Student answered: {wrong_answer}

The tutor's feedback prompt template is:
---
{eval_filled}
---

Rate this feedback template on a scale of 0-10:
- Does it give clear, actionable feedback?
- Is it kind but educational?
- Will a student learn from this?

Reply with ONLY a number from 0 to 10. Nothing else."""

        try:
            result = chat([{"role": "user", "content": meta_prompt}], temperature=0.2)
            score = float(''.join(c for c in result.strip() if c.isdigit() or c == '.'))
            score = max(0, min(10, score))
            scores.append(score)
        except:
            scores.append(5.0)

    avg = sum(scores) / len(scores) if scores else 0.0
    return round(avg, 2)

def _synthetic_failures():
    """Fallback test cases when no real student data exists yet."""
    return [
        {"topic": "math", "question": "What is the square root of 144?", "student_answer": "10", "correct": False},
        {"topic": "science", "question": "What gas do plants absorb during photosynthesis?", "student_answer": "oxygen", "correct": False},
        {"topic": "history", "question": "In what year did Pakistan gain independence?", "student_answer": "1948", "correct": False},
        {"topic": "math", "question": "Solve: 3x + 6 = 18", "student_answer": "x = 2", "correct": False},
        {"topic": "english", "question": "What is a synonym for 'happy'?", "student_answer": "sad", "correct": False},
    ]

# ── mutation (the "agent edits train.py") ─────────────────────────────────────

def mutate_prompts(current_source, student_data, log, generation):
    """
    Ask Groq to read prompts.py and suggest ONE improvement.
    Mirrors how the autoresearch agent edits train.py.
    """

    # build context from recent failures
    failures_summary = []
    for student, sdata in student_data.get("students", {}).items():
        weak = sdata.get("weak_topics", {})
        for topic, stats in weak.items():
            if stats["attempts"] > 0:
                rate = stats["wrong"] / stats["attempts"]
                if rate > 0.4:
                    failures_summary.append(f"{student} struggles with {topic} ({int(rate*100)}% wrong)")

    # include history of what already failed
    past_attempts = ""
    if log["experiments"]:
        last3 = log["experiments"][-3:]
        past_attempts = "Recent experiment results:\n" + "\n".join(
            f"  Gen {e['generation']}: score={e['score_after']} ({'KEPT' if e['kept'] else 'DISCARDED'}), change: {e['change_description']}"
            for e in last3
        )

    mutation_prompt = f"""You are an AI researcher improving a tutoring system. Your job is exactly like
karpathy's autoresearch — you modify prompt templates to improve student outcomes.

Current prompts.py file:
```python
{current_source}
```

Student failure context:
{chr(10).join(failures_summary) if failures_summary else "No student data yet — optimize for general teaching quality."}

{past_attempts}

Generation: {generation}
Your goal: modify ONE thing in prompts.py to improve avg_score (student learning outcomes).

Rules:
- You may change QUESTION_PROMPT, EVAL_PROMPT, EXPLANATION_PROMPT, or the threshold constants
- Make ONE focused change — don't rewrite everything
- The change should be motivated by the student failure data
- Keep the same variable names and file structure
- Keep the generation comment updated: # Generation: {generation}

Return ONLY the complete new prompts.py source code. No explanation. No markdown fences. Just the Python file."""

    result = chat([{"role": "user", "content": mutation_prompt}], temperature=0.9)

    # strip markdown fences if model adds them anyway
    result = result.strip()
    if result.startswith("```"):
        lines = result.split("\n")
        result = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    return result

# ── main research loop ─────────────────────────────────────────────────────────

def run_autoresearch(n_experiments=N_EXPERIMENTS):
    print("""
╔══════════════════════════════════════════════════════╗
║      AUTORESEARCH — Overnight Prompt Evolution       ║
║      Mirrors karpathy/autoresearch architecture      ║
║      Metric: avg_score (higher = better teaching)    ║
╚══════════════════════════════════════════════════════╝
""")

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: Set GROQ_API_KEY first.\n")
        return

    log          = load_log()
    student_data = load_student_data()

    # baseline score
    print("  Measuring baseline score...")
    prompts_mod   = load_prompts_module()
    baseline_score = evaluate_prompts(prompts_mod, student_data, log["generation"])
    if log["best_score"] == 0.0:
        log["best_score"] = baseline_score
    print(f"  Baseline avg_score: {baseline_score:.2f} / 10")
    print(f"  Best so far:        {log['best_score']:.2f} / 10")
    print(f"  Starting generation: {log['generation']}\n")
    print("  " + "─"*50)

    for i in range(n_experiments):
        gen = log["generation"] + 1
        print(f"\n  Experiment {i+1}/{n_experiments}  |  Generation {gen}")
        print(f"  {'─'*40}")

        # 1. backup current prompts
        backup_prompts()
        current_source = read_prompts_source()

        # 2. mutate
        print("  Mutating prompts.py...")
        try:
            new_source = mutate_prompts(current_source, student_data, log, gen)
        except Exception as e:
            print(f"  Mutation failed: {e} — skipping")
            continue

        # 3. write mutated version
        write_prompts_source(new_source)

        # 4. evaluate new version
        print("  Evaluating new prompts...")
        try:
            new_mod   = load_prompts_module()
            new_score = evaluate_prompts(new_mod, student_data, gen)
        except Exception as e:
            print(f"  Evaluation failed: {e} — discarding")
            restore_prompts()
            continue

        # 5. keep or discard (same logic as autoresearch repo)
        improved = new_score > log["best_score"]
        change_desc = _extract_change_description(current_source, new_source)

        if improved:
            log["best_score"] = new_score
            log["generation"] = gen
            status = "✓ KEPT"
            print(f"  {status}   score: {new_score:.2f}  (was {baseline_score:.2f}) ↑ IMPROVED")
        else:
            restore_prompts()
            status = "✗ DISCARDED"
            print(f"  {status}   score: {new_score:.2f}  (best: {log['best_score']:.2f}) — no improvement")

        # 6. log experiment
        log["experiments"].append({
            "timestamp":          datetime.datetime.now().isoformat(),
            "generation":         gen,
            "score_before":       baseline_score,
            "score_after":        new_score,
            "kept":               improved,
            "change_description": change_desc,
        })
        save_log(log)

        baseline_score = new_score if improved else baseline_score
        time.sleep(1)  # be gentle with rate limits

    # ── summary ──
    print(f"""
  {'═'*50}
  AUTORESEARCH COMPLETE
  Experiments run:  {n_experiments}
  Best avg_score:   {log['best_score']:.2f} / 10
  Final generation: {log['generation']}
  Kept experiments: {sum(1 for e in log['experiments'] if e['kept'])}
  
  Updated prompts saved to: prompts.py
  Full log saved to:        autoresearch_log.json
  {'═'*50}
""")

def _extract_change_description(old_source, new_source):
    """Quick summary of what changed between two prompts.py versions."""
    old_lines = set(old_source.splitlines())
    new_lines = set(new_source.splitlines())
    added = [l.strip() for l in (new_lines - old_lines) if l.strip() and not l.strip().startswith("#")]
    if added:
        return added[0][:120]
    return "minor rephrasing"

def print_log():
    """Print a summary of all experiments."""
    log = load_log()
    if not log["experiments"]:
        print("No experiments run yet.")
        return
    print(f"\n  Autoresearch Log — {len(log['experiments'])} experiments\n")
    print(f"  {'Gen':<5} {'Score':<8} {'Result':<12} Change")
    print("  " + "─"*60)
    for e in log["experiments"]:
        kept = "✓ KEPT" if e["kept"] else "✗ DISCARDED"
        print(f"  {e['generation']:<5} {e['score_after']:<8.2f} {kept:<12} {e['change_description'][:40]}")
    print(f"\n  Best score: {log['best_score']:.2f} / 10  |  Generation: {log['generation']}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "log":
        print_log()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else N_EXPERIMENTS
        run_autoresearch(n)