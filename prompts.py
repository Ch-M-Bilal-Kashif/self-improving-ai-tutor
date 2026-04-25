# prompts.py — THE FILE THE AGENT EDITS
# This is the equivalent of train.py in the autoresearch repo.
# The autoresearch agent modifies ONLY this file overnight.
# Every prompt template here is fair game: wording, structure, style, strategy.
# The metric is student_score (avg correct rate) — higher is better.
# Do NOT modify tutor.py or autoresearch.py manually.

# —— Generation: 3 ————
PROMPT_VERSION = 2

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
ANSWER: 4"""

# Controls how wrong answers are evaluated and feedback is given
EVAL_PROMPT = """You are a strict but kind teacher evaluating a student's answer.

Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {student_answer}

Reply in this exact format:
SCORE: [0-10]
CORRECT: [yes/no]
FEEDBACK: [1-2 sentences. If wrong, explain WHY clearly and give the right answer simply.]"""

# Controls how concepts are explained after failures
EXPLANATION_PROMPT = """Explain "{topic}" to a school student in simple, clear language.
{weakness_context}
Use a real-world example or analogy. Keep it under 150 words. Be encouraging and friendly."""

# Controls the weakness context injected into prompts
WEAKNESS_CONTEXT_TEMPLATE = "The student has previously struggled with: {topics}. Focus on those weak points."

# Difficulty thresholds
EASY_THRESHOLD = 0       # attempts before bumping to medium
MEDIUM_THRESHOLD = 3     # attempts before bumping to hard
HARD_WRONG_RATE = 0.5    # if wrong rate on topic > this, use hard difficulty