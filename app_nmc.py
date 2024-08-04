import os
import streamlit as st


import torch
from transformers import pipeline
from datasets import load_dataset
import evaluate

# Kiểm tra xem torch đã được cài đặt và có thể sử dụng GPU hay không
if not torch.cuda.is_available():
    st.warning("CUDA is not available. The application will run on CPU.")

# Khởi tạo mô hình hỏi-đáp đã huấn luyện
PIPELINE_NAME = 'question-answering'
MODEL_NAME = 'NMC-29092004/distilbert-finetuned-squadv2-1'

device = 0 if torch.cuda.is_available() else -1  # Sử dụng GPU nếu có, nếu không sẽ dùng CPU
pipe = pipeline(PIPELINE_NAME, model=MODEL_NAME, device=device)

# Hàm lấy câu trả lời tốt nhất
def get_best_answer(input_question):
    input_quest_embedding = get_embeddings([input_question])
    input_quest_embedding = input_quest_embedding.cpu().detach().numpy()

    TOP_K = 5
    scores, samples = embeddings_dataset.get_nearest_examples(
        EMBEDDING_COLUMN, input_quest_embedding, k=TOP_K
    )

    best_answer = None
    best_score = float('inf')  # Khởi tạo với điểm số vô cùng lớn để so sánh

    for idx, score in enumerate(scores):
        question = samples["question"][idx]
        context = samples["context"][idx]
        answer = pipe(
            question=input_question,
            context=context
        )
        if score < best_score:  # So sánh điểm số, điểm càng thấp càng tốt
            best_score = score
            best_answer = answer

    return best_answer

# Giao diện Streamlit
st.title("Demo Hỏi-Đáp")

input_question = st.text_input("Nhập câu hỏi của bạn:", value="When did Beyonce start becoming popular?")

# Sử dụng hàm get_best_answer để tìm câu trả lời tốt nhất
if input_question:
    best_answer = get_best_answer(input_question)
    st.write("### Câu trả lời tốt nhất:")
    st.write(f"**Answer**: {best_answer['answer']}")
    st.write(f"**Score**: {best_answer['score']}")
