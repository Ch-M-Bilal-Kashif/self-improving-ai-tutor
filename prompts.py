# prompts.py — THE FILE THE AGENT EDITS
# Autoresearch prompt system

# —— Generation: 3 ————
PROMPT_VERSION = 3

# Controls how questions are generated
QUESTION_PROMPT = """You are a tutor generating a quiz question.

Topic: {topic}
Difficulty: {difficulty}
{weakness_context}

Generate ONE clear question suitable for a school student that targets their specific weaknesses.

Then on a new line write "ANSWER:" followed by the correct answer.

Keep it concise.

Example:
What is 2 + 2?
ANSWER: 4
"""

# Controls evaluation of answers
EVAL_PROMPT = """You are a strict but kind teacher evaluating a student's answer.

Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {student_answer}

Reply in this exact format:

SCORE: [0-10]
CORRECT: [yes/no]
FEEDBACK: [1-2 sentences explaining clearly if wrong]
"""

# Controls explanation after failure
EXPLANATION_PROMPT = """Explain "{topic}" to a school student in simple, clear language.

{weakness_context}

Use a real-world example or analogy.
Keep it under 150 words.
Be encouraging and friendly.
"""

# Weakness injection template
WEAKNESS_CONTEXT_TEMPLATE = (
    "The student has previously struggled with: {topics}. "
    "Focus on those weak points."
)

# Difficulty thresholds (tuned properly)
EASY_THRESHOLD = 2
MEDIUM_THRESHOLD = 5
HARD_WRONG_RATE = 0.5
