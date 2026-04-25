"""
app.py — Streamlit UI for AI Tutor + Autoresearch
Run with: streamlit run app.py
"""

import os
import json
import time
import datetime
import threading
import importlib.util
from pathlib import Path



import streamlit as st
import openai

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Tutor — Self-Improving",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, .big-title {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background: #0a0a0f;
    color: #e8e6f0;
}

section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}

.metric-card {
    background: #12121f;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}

.metric-card .label {
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b6b8a;
    margin-bottom: 6px;
}

.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #c8b8ff;
}

.metric-card .sub {
    font-size: 12px;
    color: #4a4a6a;
    margin-top: 4px;
}

.question-box {
    background: #12121f;
    border: 1px solid #2a2a4a;
    border-left: 3px solid #8b5cf6;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    font-size: 1.1rem;
    color: #e8e6f0;
    margin: 1rem 0;
    line-height: 1.6;
}

.feedback-correct {
    background: #0d1f0d;
    border: 1px solid #1a3a1a;
    border-left: 3px solid #22c55e;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #86efac;
    margin: 0.8rem 0;
}

.feedback-wrong {
    background: #1f0d0d;
    border: 1px solid #3a1a1a;
    border-left: 3px solid #ef4444;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #fca5a5;
    margin: 0.8rem 0;
}

.weak-bar-wrap {
    background: #12121f;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin: 0.4rem 0;
    border: 1px solid #1e1e2e;
}

.weak-topic-name {
    font-size: 13px;
    font-weight: 500;
    color: #c8b8ff;
    margin-bottom: 4px;
}

.experiment-row {
    background: #12121f;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 0.3rem 0;
    display: flex;
    gap: 12px;
    align-items: center;
    font-size: 13px;
}

.badge-kept {
    background: #052e16;
    color: #4ade80;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 500;
}

.badge-disc {
    background: #2d0a0a;
    color: #f87171;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 500;
}

.version-tag {
    display: inline-block;
    background: #1a0f2e;
    border: 1px solid #4c1d95;
    color: #a78bfa;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 99px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.05em;
}

div[data-testid="stButton"] button {
    background: #1a0f2e !important;
    border: 1px solid #4c1d95 !important;
    color: #c4b5fd !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}

div[data-testid="stButton"] button:hover {
    background: #2d1b69 !important;
    border-color: #7c3aed !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select,
div[data-testid="stTextArea"] textarea {
    background: #12121f !important;
    border: 1px solid #2a2a4a !important;
    color: #e8e6f0 !important;
    border-radius: 8px !important;
}

.stProgress > div > div > div {
    background: #7c3aed !important;
}

hr {
    border-color: #1e1e2e !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1a;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: #12121f;
    border-radius: 8px;
    color: #6b6b8a;
    font-family: 'DM Sans', sans-serif;
}

.stTabs [aria-selected="true"] {
    background: #1a0f2e !important;
    color: #c4b5fd !important;
}

.stAlert {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
DATA_FILE = "student_data.json"
LOG_FILE  = "autoresearch_log.json"
TOPICS    = ["math", "science", "history", "english", "urdu", "general knowledge"]

# ── groq client ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY", "")
    )
def chat(messages, temperature=0.7):
    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=temperature,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()

st.secrets["GROQ_API_KEY"]

# ── data helpers ──────────────────────────────────────────────────────────────
def load_data():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"students": {}, "overnight_insights": [], "model_version": 1}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_log():
    if Path(LOG_FILE).exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return {"experiments": [], "best_score": 0.0, "generation": 1}

def load_prompts():
    try:
        spec = importlib.util.spec_from_file_location("prompts", "prompts.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except:
        return None

def get_weak_topics(data, name):
    if name not in data["students"]:
        return []
    weak = data["students"][name]["weak_topics"]
    return sorted(
        [(t, v["wrong"] / max(v["attempts"], 1), v["attempts"]) for t, v in weak.items()],
        key=lambda x: -x[1]
    )

def record_attempt(data, name, topic, question, answer, correct, score, feedback):
    if name not in data["students"]:
        data["students"][name] = {"attempts": [], "weak_topics": {}}
    data["students"][name]["attempts"].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "topic": topic, "question": question,
        "student_answer": answer, "correct": correct,
        "score": score, "feedback": feedback
    })
    weak = data["students"][name]["weak_topics"]
    if topic not in weak:
        weak[topic] = {"attempts": 0, "wrong": 0}
    weak[topic]["attempts"] += 1
    if not correct:
        weak[topic]["wrong"] += 1
    save_data(data)

def all_students_stats(data):
    stats = []
    for name, sdata in data["students"].items():
        attempts = sdata["attempts"]
        if not attempts:
            continue
        correct = sum(1 for a in attempts if a["correct"])
        stats.append({
            "name": name,
            "total": len(attempts),
            "correct": correct,
            "rate": correct / len(attempts),
            "weak": sorted(sdata["weak_topics"].items(),
                           key=lambda x: x[1]["wrong"] / max(x[1]["attempts"], 1),
                           reverse=True)
        })
    return stats

# ── AI functions ──────────────────────────────────────────────────────────────
def generate_question(topic, difficulty, weak_topics, prompts_mod):
    weakness_context = ""
    if weak_topics:
        template = getattr(prompts_mod, "WEAKNESS_CONTEXT_TEMPLATE",
                           "The student has previously struggled with: {topics}.")
        weakness_context = template.format(topics=", ".join(weak_topics))

    prompt_template = getattr(prompts_mod, "QUESTION_PROMPT", """
You are a tutor generating a quiz question.
Topic: {topic}
Difficulty: {difficulty}
{weakness_context}
Generate ONE clear question. Then write "ANSWER:" followed by the correct answer.""")

    filled = prompt_template.format(
        topic=topic, difficulty=difficulty, weakness_context=weakness_context)

    result = chat([{"role": "user", "content": filled}], temperature=0.8)

    if "ANSWER:" in result:
        parts = result.split("ANSWER:")
        return parts[0].strip(), parts[1].strip()
    return result.strip(), "See explanation"

def evaluate_answer(question, correct_answer, student_answer, prompts_mod):
    template = getattr(prompts_mod, "EVALUATION_PROMPT", """
You are a teacher evaluating a student's answer.
Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {student_answer}
Reply:
SCORE: [0-10]
CORRECT: [yes/no]
FEEDBACK: [1-2 sentences]""")

    filled = template.format(
        question=question,
        correct_answer=correct_answer,
        student_answer=student_answer
    )

    result = chat([{"role": "user", "content": filled}], temperature=0.3)
    score, correct, feedback = 5, False, result

    for line in result.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = int(''.join(c for c in line.replace("SCORE:", "").strip() if c.isdigit()))
            except:
                pass
        elif line.startswith("CORRECT:"):
            correct = "yes" in line.lower()
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()

    return score, correct, feedback

def explain_concept(topic, weak_topics, prompts_mod):
    weakness_context = ""
    if weak_topics:
        template = getattr(prompts_mod, "WEAKNESS_CONTEXT_TEMPLATE",
                           "The student has struggled with: {topics}.")
        weakness_context = template.format(topics=", ".join(weak_topics))

    template = getattr(prompts_mod, "EXPLANATION_PROMPT", """
Explain "{topic}" to a school student simply.
{weakness_context}
Use a real-world example. Under 150 words. Be encouraging.""")

    filled = template.format(topic=topic, weakness_context=weakness_context)
    return chat([{"role": "user", "content": filled}])

# ── sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(data, log, prompts_mod):
    with st.sidebar:
        st.markdown("### 🧠 AI Tutor")
        st.markdown(f'<span class="version-tag">v{data["model_version"]} · Gen {log["generation"]}</span>',
                    unsafe_allow_html=True)
        st.markdown("---")

        mode = st.radio("View", ["👨‍🎓 Student", "👩‍🏫 Teacher"], label_visibility="collapsed")
        st.markdown("---")

        if mode == "👨‍🎓 Student":
            name = st.text_input("Your name", placeholder="Enter your name")
            topic = st.selectbox("Topic", TOPICS)
            return mode, name, topic
        else:
            return mode, None, None

# ── student view ──────────────────────────────────────────────────────────────
def render_student(data, prompts_mod, name, topic):
    if not name:
        st.markdown("## 👋 Welcome!")
        st.info("Enter your name in the sidebar to get started.")
        return

    weak = get_weak_topics(data, name)
    weak_names = [t for t, _, _ in weak[:3]]
    attempts = data["students"].get(name, {}).get("attempts", [])
    total    = len(attempts)
    correct  = sum(1 for a in attempts if a["correct"])

    # header
    st.markdown(f"## Hey, {name}! 👋")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="label">Questions</div><div class="value">{total}</div><div class="sub">attempted</div></div>', unsafe_allow_html=True)
    with col2:
        rate = int(correct / total * 100) if total else 0
        st.markdown(f'<div class="metric-card"><div class="label">Accuracy</div><div class="value">{rate}%</div><div class="sub">correct</div></div>', unsafe_allow_html=True)
    with col3:
        avg_score = sum(a["score"] for a in attempts) / total if total else 0
        st.markdown(f'<div class="metric-card"><div class="label">Avg Score</div><div class="value">{avg_score:.1f}</div><div class="sub">out of 10</div></div>', unsafe_allow_html=True)
    with col4:
        streak = 0
        for a in reversed(attempts):
            if a["correct"]: streak += 1
            else: break
        st.markdown(f'<div class="metric-card"><div class="label">Streak</div><div class="value">{streak}</div><div class="sub">in a row ✓</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📝 Quiz", "📖 Explain It", "📊 My Progress"])

    # ── quiz tab ──
    with tab1:
        st.markdown(f"### Quiz: {topic.title()}")

        # auto-adjust difficulty
        if weak_names and topic in weak_names:
            difficulty = "hard"
            st.warning(f"⚠️ You've struggled with **{topic}** before. Bringing the heat.")
        elif total > 10:
            difficulty = "medium"
        else:
            difficulty = "easy"

        if st.button("🎲 Generate Question", use_container_width=True):
            with st.spinner("Generating question..."):
                q, ans = generate_question(topic, difficulty, weak_names, prompts_mod)
            st.session_state["current_q"]   = q
            st.session_state["current_ans"] = ans
            st.session_state["topic"]       = topic
            st.session_state["answered"]    = False
            st.session_state["feedback"]    = None

        if "current_q" in st.session_state and not st.session_state.get("answered"):
            st.markdown(f'<div class="question-box">❓ {st.session_state["current_q"]}</div>',
                        unsafe_allow_html=True)
            student_answer = st.text_input("Your answer:", key="answer_input")

            if st.button("✅ Submit Answer", use_container_width=True):
                if student_answer.strip():
                    with st.spinner("Evaluating..."):
                        score, correct, feedback = evaluate_answer(
                            st.session_state["current_q"],
                            st.session_state["current_ans"],
                            student_answer,
                            prompts_mod
                        )
                    record_attempt(data, name, st.session_state["topic"],
                                   st.session_state["current_q"], student_answer,
                                   correct, score, feedback)
                    st.session_state["answered"]  = True
                    st.session_state["feedback"]  = feedback
                    st.session_state["correct"]   = correct
                    st.session_state["score"]     = score
                    st.rerun()

        if st.session_state.get("answered"):
            correct = st.session_state["correct"]
            score   = st.session_state["score"]
            feedback = st.session_state["feedback"]

            if correct:
                st.markdown(f'<div class="feedback-correct">✓ Correct! Score: {score}/10<br>{feedback}</div>',
                            unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<div class="feedback-wrong">✗ Not quite. Score: {score}/10<br>{feedback}</div>',
                            unsafe_allow_html=True)

                if st.button("💡 Explain this topic", use_container_width=True):
                    with st.spinner("Generating explanation..."):
                        explanation = explain_concept(
                            st.session_state["topic"], weak_names, prompts_mod)
                    st.info(explanation)

    # ── explain tab ──
    with tab2:
        st.markdown("### 📖 Learn a Topic")
        exp_topic = st.selectbox("Choose topic to explain", TOPICS, key="exp_topic")
        if st.button("✨ Explain it to me", use_container_width=True):
            with st.spinner("Thinking..."):
                explanation = explain_concept(exp_topic, weak_names, prompts_mod)
            st.info(explanation)

    # ── progress tab ──
    with tab3:
        st.markdown("### 📊 Your Weak Areas")
        if not weak:
            st.info("No data yet. Take some quizzes first!")
        else:
            for t, rate, attempts_count in weak:
                pct = int(rate * 100)
                color = "#ef4444" if pct > 60 else "#f59e0b" if pct > 30 else "#22c55e"
                st.markdown(f"""
                <div class="weak-bar-wrap">
                    <div class="weak-topic-name">{t.title()} — {pct}% wrong ({attempts_count} attempts)</div>
                </div>""", unsafe_allow_html=True)
                st.progress(rate, text="")

        st.markdown("### 📜 Recent Attempts")
        recent = list(reversed(attempts[-10:]))
        for a in recent:
            icon = "✓" if a["correct"] else "✗"
            color = "green" if a["correct"] else "red"
            st.markdown(f":{color}[{icon}] **{a['topic'].title()}** — {a['question'][:60]}... *(score: {a['score']}/10)*")

# ── teacher view ──────────────────────────────────────────────────────────────
def render_teacher(data, log, prompts_mod):
    st.markdown("## 👩‍🏫 Teacher Dashboard")

    stats = all_students_stats(data)
    total_students  = len(stats)
    total_questions = sum(s["total"] for s in stats)
    avg_accuracy    = sum(s["rate"] for s in stats) / len(stats) if stats else 0
    gen             = log["generation"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="label">Students</div><div class="value">{total_students}</div><div class="sub">active</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="label">Questions</div><div class="value">{total_questions}</div><div class="sub">answered</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="label">Avg Accuracy</div><div class="value">{int(avg_accuracy*100)}%</div><div class="sub">class wide</div></div>', unsafe_allow_html=True)
    with col4:
        best = log.get("best_score", 0)
        st.markdown(f'<div class="metric-card"><div class="label">AI Score</div><div class="value">{best:.1f}</div><div class="sub">teaching quality</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["👥 Students", "🔬 Autoresearch Log", "⚙️ Run Autoresearch"])

    # ── students tab ──
    with tab1:
        if not stats:
            st.info("No students yet. Share the app with your students!")
        else:
            for s in sorted(stats, key=lambda x: x["rate"]):
                pct   = int(s["rate"] * 100)
                color = "🔴" if pct < 50 else "🟡" if pct < 75 else "🟢"
                with st.expander(f"{color} {s['name']} — {pct}% accuracy ({s['total']} questions)"):
                    st.progress(s["rate"])
                    if s["weak"]:
                        st.markdown("**Weakest topics:**")
                        for topic, vals in s["weak"][:3]:
                            rate = vals["wrong"] / max(vals["attempts"], 1)
                            st.markdown(f"- {topic.title()}: {int(rate*100)}% wrong")

    # ── autoresearch log tab ──
    with tab2:
        experiments = log.get("experiments", [])
        if not experiments:
            st.info("No autoresearch runs yet. Go to the 'Run Autoresearch' tab.")
        else:
            st.markdown(f"**{len(experiments)} experiments run · Best score: {log['best_score']:.2f}/10 · Generation {log['generation']}**")
            st.markdown("")
            for e in reversed(experiments[-20:]):
                kept  = e["kept"]
                badge = '<span class="badge-kept">✓ KEPT</span>' if kept else '<span class="badge-disc">✗ DISCARDED</span>'
                delta = e["score_after"] - e["score_before"]
                arrow = f"↑ +{delta:.2f}" if delta > 0 else f"↓ {delta:.2f}" if delta < 0 else "→ 0"
                st.markdown(f"""
                <div class="experiment-row">
                    {badge}
                    <span style="color:#6b6b8a;font-size:12px">Gen {e['generation']}</span>
                    <span style="color:#c8b8ff">score: {e['score_after']:.2f}</span>
                    <span style="color:#6b6b8a">{arrow}</span>
                    <span style="color:#4a4a6a;flex:1">{e['change_description'][:60]}</span>
                </div>""", unsafe_allow_html=True)

    # ── run autoresearch tab ──
    with tab3:
        st.markdown("### 🔬 Run Overnight Autoresearch")
        st.markdown("""
This runs the autoresearch loop — exactly like Karpathy's repo but for prompts instead of model weights.

The agent will:
1. Read `prompts.py` (your current teaching prompts)
2. Propose an improvement based on student failures
3. Test it against real student data
4. Keep it if better, discard if worse
5. Repeat N times
        """)

        n_exp = st.slider("Number of experiments", 1, 20, 5)

        if st.button("🚀 Start Autoresearch Now", use_container_width=True):
            # import and run autoresearch inline
            progress_bar = st.progress(0)
            status_text  = st.empty()
            results_area = st.empty()

            try:
                import autoresearch as ar

                student_data = load_data()
                ar_log       = load_log()

                status_text.text("Measuring baseline score...")
                pm = load_prompts()
                baseline = ar.evaluate_prompts(pm, student_data, ar_log["generation"])
                if ar_log["best_score"] == 0.0:
                    ar_log["best_score"] = baseline

                results = []

                for i in range(n_exp):
                    progress_bar.progress((i) / n_exp)
                    status_text.text(f"Experiment {i+1}/{n_exp} — mutating prompts...")

                    gen = ar_log["generation"] + 1
                    ar.backup_prompts()
                    current_source = ar.read_prompts_source()

                    try:
                        new_source = ar.mutate_prompts(current_source, student_data, ar_log, gen)
                        ar.write_prompts_source(new_source)

                        status_text.text(f"Experiment {i+1}/{n_exp} — evaluating...")
                        new_mod   = ar.load_prompts_module()
                        new_score = ar.evaluate_prompts(new_mod, student_data, gen)

                        improved = new_score > ar_log["best_score"]
                        change   = ar._extract_change_description(current_source, new_source)

                        if improved:
                            ar_log["best_score"] = new_score
                            ar_log["generation"] = gen
                        else:
                            ar.restore_prompts()

                        entry = {
                            "timestamp":          datetime.datetime.now().isoformat(),
                            "generation":         gen,
                            "score_before":       baseline,
                            "score_after":        new_score,
                            "kept":               improved,
                            "change_description": change,
                        }
                        ar_log["experiments"].append(entry)
                        ar.save_log(ar_log)
                        results.append(entry)
                        baseline = new_score if improved else baseline

                    except Exception as e:
                        ar.restore_prompts()
                        results.append({"error": str(e), "generation": gen})

                    time.sleep(0.5)

                progress_bar.progress(1.0)
                status_text.text("✅ Autoresearch complete!")

                # update model version in student data
                sd = load_data()
                sd["model_version"] = ar_log["generation"]
                save_data(sd)

                kept = sum(1 for r in results if r.get("kept"))
                st.success(f"Done! {kept}/{n_exp} experiments improved the model. Best score: {ar_log['best_score']:.2f}/10")
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("**Or run overnight from terminal:**")
        st.code("python autoresearch.py 20", language="bash")
        st.markdown("*(runs 20 experiments, saves to autoresearch_log.json)*")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ Set your GROQ_API_KEY environment variable or add it to Streamlit secrets.")
        st.code("export GROQ_API_KEY=your_key_here", language="bash")
        st.stop()

    data       = load_data()
    log        = load_log()
    prompts_mod = load_prompts()

    if prompts_mod is None:
        st.error("⚠️ prompts.py not found. Make sure it's in the same directory as app.py.")
        st.stop()

    # init session state
    for key in ["current_q", "current_ans", "answered", "feedback", "correct", "score", "topic"]:
        if key not in st.session_state:
            st.session_state[key] = None

    mode, name, topic = render_sidebar(data, log, prompts_mod)

    if mode == "👨‍🎓 Student":
        render_student(data, prompts_mod, name, topic or TOPICS[0])
    else:
        render_teacher(data, log, prompts_mod)

if __name__ == "__main__":
    main()
