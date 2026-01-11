import streamlit as st
import time
import os

from knowledge.pdf_loader import process_pdf
from knowledge.vector_store import build_vector_store
from knowledge.query import ask_pdf
from rl_policy import (
    get_teaching_style,
    update_state,
    save_student_state
)

st.set_page_config(page_title="Self-Improving AI Tutor")
st.title("🧠 Self-Improving AI Tutor")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Processing PDF...")
    docs = process_pdf("temp.pdf")
    build_vector_store(docs)
    st.success("PDF successfully indexed!")

question = st.text_input("Ask a question from the PDF")

if question:
    start_time = time.time()

    style, state = get_teaching_style()
    answer = ask_pdf(question, style)

    st.markdown(f"### 🎓 Teaching Style: {style}")
    st.write(answer)

    response_time = time.time() - start_time

    col1, col2 = st.columns(2)
    reward = None

    if col1.button("✅ Understood"):
        reward = 1
    if col2.button("❌ Not Clear"):
        reward = -1

    if reward is not None:
        new_state = update_state(state, reward, response_time)
        save_student_state(new_state)

        with open("data/feedback.csv", "a") as f:
            f.write(f"{question},{reward},{response_time}\n")

        st.success("Tutor adapted based on feedback!")
