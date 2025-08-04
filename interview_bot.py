import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal
from uuid import uuid4

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

class Evaluation(BaseModel):
    grade: Literal["strong", "weak"] = Field(description="Evaluate answer strength")
    feedback: str = Field(description="Suggestions to improve")

evaluator = llm.with_structured_output(Evaluation)

# Session state setup
if "history" not in st.session_state:
    st.session_state.history = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "profession" not in st.session_state:
    st.session_state.profession = ""

st.title("🧠 AI Interview Coach (Text-Based)")

# Upload resume and profession
with st.sidebar:
    st.header("Candidate Info")
    uploaded_file = st.file_uploader("Upload your resume (TXT or PDF)", type=["txt", "pdf"])
    st.session_state.profession = st.text_input("Your profession", value=st.session_state.profession)

    if uploaded_file:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        st.session_state.resume_text = content

# Generate an interview question
def generate_question():
    resume = st.session_state.resume_text or "No resume provided"
    profession = st.session_state.profession or "a software engineer"
    msg = llm.invoke(
        f"As an interviewer for {profession}, generate a behavioral interview question based on this resume:\n{resume}"
    )
    return msg.content.strip()

# Evaluate answer
def evaluate_answer(question, answer):
    eval_input = f"Evaluate the following answer using STAR format.\nQuestion: {question}\nAnswer: {answer}"
    result = evaluator.invoke(eval_input)
    return result

# Main interaction
if st.button("Start Interview" if st.session_state.question_count == 0 else "Next Question"):
    question = generate_question()
    st.session_state.question_count += 1
    st.session_state.history.append({"question": question, "answer": "", "feedback": "", "grade": ""})

# Show last question if exists
if st.session_state.history:
    current = st.session_state.history[-1]
    st.subheader(f"💬 Question {st.session_state.question_count}")
    st.write(current["question"])

    answer_input = st.text_area("🗣️ Your Answer", value=current["answer"], key=str(uuid4()))

    if st.button("Submit Answer"):
        result = evaluate_answer(current["question"], answer_input)
        current["answer"] = answer_input
        current["grade"] = result.grade
        current["feedback"] = result.feedback
        st.success(f"✅ Grade: {result.grade}")
        st.info(f"💡 Feedback: {result.feedback}")

# Optionally show chat history
if st.checkbox("Show Full History"):
    for idx, item in enumerate(st.session_state.history):
        st.write(f"### Q{idx+1}: {item['question']}")
        st.write(f"Answer: {item['answer']}")
        st.write(f"Grade: {item['grade']}")
        st.write(f"Feedback: {item['feedback']}")
