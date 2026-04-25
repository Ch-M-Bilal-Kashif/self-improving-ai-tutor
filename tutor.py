import os
import json
import datetime
from pathlib import Path
import openai

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

DATA_FILE = "student_data.json"




# ── data layer ────────────────────────────────────────────────────────────────

def load_data():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"students": {}, "overnight_insights": [], "model_version": 1}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def record_attempt(data, student_name, topic, question, student_answer, correct, score):
    if student_name not in data["students"]:
        data["students"][student_name] = {"attempts": [], "weak_topics": {}}
    
    data["students"][student_name]["attempts"].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "topic": topic,
        "question": question,
        "student_answer": student_answer,
        "correct": correct,
        "score": score
    })

    # track weak topics
    weak = data["students"][student_name]["weak_topics"]
    if topic not in weak:
        weak[topic] = {"attempts": 0, "wrong": 0}
    weak[topic]["attempts"] += 1
    if not correct:
        weak[topic]["wrong"] += 1

    save_data(data)

def get_weak_topics(data, student_name):
    if student_name not in data["students"]:
        return []
    weak = data["students"][student_name]["weak_topics"]
    return sorted(
        [(t, v["wrong"] / max(v["attempts"], 1)) for t, v in weak.items()],
        key=lambda x: -x[1]
    )

# ── groq calls ────────────────────────────────────────────────────────────────

def chat(messages, temperature=0.7):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def generate_question(topic, difficulty="medium", previous_mistakes=None):
    context = ""
    if previous_mistakes:
        context = f"The student has previously struggled with: {', '.join(previous_mistakes)}. Focus on those weak points."
    
    prompt = f"""You are a tutor generating a quiz question.
Topic: {topic}
Difficulty: {difficulty}
{context}

Generate ONE clear question. Then on a new line write "ANSWER:" followed by the correct answer.
Keep it concise. Example format:
What is 2 + 2?
ANSWER: 4"""

    result = chat([{"role": "user", "content": prompt}], temperature=0.8)
    
    if "ANSWER:" in result:
        parts = result.split("ANSWER:")
        question = parts[0].strip()
        answer = parts[1].strip()
    else:
        question = result
        answer = "See explanation"
    
    return question, answer

def evaluate_answer(question, correct_answer, student_answer, topic):
    prompt = f"""You are a strict but kind teacher evaluating a student's answer.

Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {student_answer}

Reply in this exact format:
SCORE: [0-10]
CORRECT: [yes/no]
FEEDBACK: [1-2 sentences of helpful feedback. If wrong, explain WHY and give the right answer simply.]"""

    result = chat([{"role": "user", "content": prompt}], temperature=0.3)
    
    score = 5
    correct = False
    feedback = result

    for line in result.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = int(line.replace("SCORE:", "").strip())
            except:
                pass
        elif line.startswith("CORRECT:"):
            correct = "yes" in line.lower()
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()

    return score, correct, feedback

def explain_concept(topic, student_name, weak_subtopics=None):
    context = ""
    if weak_subtopics:
        context = f"Focus especially on these areas where {student_name} has struggled: {', '.join(weak_subtopics)}"
    
    prompt = f"""Explain "{topic}" to a student in simple, clear language.
{context}
Use an analogy or real-world example. Keep it under 150 words. Be encouraging."""

    return chat([{"role": "user", "content": prompt}])

def overnight_analysis(data):
    """The autoresearch core — runs analysis on all student failures and generates improved teaching strategies."""
    print("\n" + "="*55)
    print("  OVERNIGHT AUTORESEARCH — Analyzing student failures...")
    print("="*55)

    all_failures = []
    for student, sdata in data["students"].items():
        for attempt in sdata["attempts"]:
            if not attempt["correct"]:
                all_failures.append({
                    "student": student,
                    "topic": attempt["topic"],
                    "question": attempt["question"],
                    "wrong_answer": attempt["student_answer"]
                })

    if not all_failures:
        print("  No failures to analyze yet. Keep teaching!")
        return

    failure_summary = json.dumps(all_failures[-30:], indent=2)  # last 30 failures

    prompt = f"""You are an AI research system analyzing student failure patterns to improve teaching.

Here are recent student mistakes:
{failure_summary}

Analyze these failures and produce:
1. TOP 3 MISCONCEPTIONS: What concepts are students most confused about?
2. TEACHING STRATEGY: How should the tutor explain these better tomorrow?
3. RECOMMENDED QUESTIONS: 3 new questions that specifically target these weaknesses.

Be specific and actionable. This will update the tutor's strategy for tomorrow."""

    print("\n  Sending failures to Groq for analysis...\n")
    insight = chat([{"role": "user", "content": prompt}], temperature=0.5)

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "failures_analyzed": len(all_failures),
        "insight": insight,
        "model_version": data["model_version"] + 1
    }

    data["overnight_insights"].append(entry)
    data["model_version"] += 1
    save_data(data)

    print(insight)
    print(f"\n  Model upgraded to version {data['model_version']}")
    print("="*55)

def get_latest_strategy(data):
    if data["overnight_insights"]:
        return data["overnight_insights"][-1]["insight"]
    return None

# ── UI ────────────────────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║                                                      ║
║        AI TUTOR  —  Self-Improving Every Night       ║
║        Powered by Groq + Autoresearch                ║
║                                                      ║
╚══════════════════════════════════════════════════════╝""")

def print_menu():
    print("""
┌─────────────────────────────────────────┐
│  1. Start a quiz session                │
│  2. Explain a topic to me               │
│  3. See my weak topics                  │
│  4. Run overnight analysis              │
│  5. View latest AI teaching strategy    │
│  6. Exit                                │
└─────────────────────────────────────────┘""")

def quiz_session(data, student_name):
    print(f"\n  Topics: math, science, history, english, urdu, general")
    topic = input("  Enter topic: ").strip().lower()
    
    # use overnight insights to adjust difficulty
    weak = get_weak_topics(data, student_name)
    weak_names = [t for t, _ in weak[:3]]
    
    difficulty = "easy"
    if weak and topic in weak_names:
        difficulty = "hard"
        print(f"  ⚠  You've struggled with {topic} before. Targeting your weak points.")
    elif len(data["students"].get(student_name, {}).get("attempts", [])) > 5:
        difficulty = "medium"

    print(f"\n  Generating {difficulty} question on {topic}...\n")
    
    question, correct_answer = generate_question(topic, difficulty, weak_names)
    
    print(f"  Q: {question}\n")
    student_answer = input("  Your answer: ").strip()
    
    print("\n  Evaluating...")
    score, correct, feedback = evaluate_answer(question, correct_answer, student_answer, topic)
    
    record_attempt(data, student_name, topic, question, student_answer, correct, score)
    
    status = "✓ CORRECT" if correct else "✗ INCORRECT"
    print(f"\n  {status}  |  Score: {score}/10")
    print(f"  {feedback}")
    
    # offer explanation if wrong
    if not correct:
        explain = input("\n  Want a full explanation? (y/n): ").strip().lower()
        if explain == "y":
            print("\n" + explain_concept(topic, student_name, weak_names))

def show_weak_topics(data, student_name):
    weak = get_weak_topics(data, student_name)
    if not weak:
        print("\n  No data yet. Take some quizzes first!")
        return
    print(f"\n  Weak topics for {student_name}:")
    print("  " + "-"*35)
    for topic, rate in weak:
        bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
        print(f"  {topic:<15} {bar}  {int(rate*100)}% wrong")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print_banner()
    
    if not os.environ.get("GROQ_API_KEY"):
        print("\n  ERROR: Set your GROQ_API_KEY environment variable first.")
        print("  export GROQ_API_KEY=your_key_here\n")
        return

    data = load_data()
    
    student_name = input("\n  Enter your name: ").strip().title()
    print(f"\n  Welcome, {student_name}! Model version: v{data['model_version']}")

    # show if there's an updated strategy from overnight
    strategy = get_latest_strategy(data)
    if strategy:
        print(f"\n  AI upgraded overnight based on student failures (v{data['model_version']})")

    while True:
        print_menu()
        choice = input("  Choose: ").strip()

        if choice == "1":
            quiz_session(data, student_name)
        elif choice == "2":
            topic = input("\n  Which topic to explain? ").strip()
            weak = [t for t, _ in get_weak_topics(data, student_name)[:3]]
            print(f"\n{explain_concept(topic, student_name, weak)}")
        elif choice == "3":
            show_weak_topics(data, student_name)
        elif choice == "4":
            overnight_analysis(data)
        elif choice == "5":
            strategy = get_latest_strategy(data)
            if strategy:
                print(f"\n  Latest AI Teaching Strategy (v{data['model_version']}):\n")
                print(strategy)
            else:
                print("\n  No overnight analysis run yet. Choose option 4 first.")
        elif choice == "6":
            print(f"\n  Goodbye, {student_name}! Keep learning.\n")
            break
        else:
            print("\n  Invalid choice.")

if __name__ == "__main__":
    main()
