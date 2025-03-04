import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from datasets import load_dataset
import numpy as np
import faiss

# Thiết lập thiết bị (GPU nếu có, nếu không sẽ dùng CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Tải dataset và lọc dữ liệu
DATASET_NAME = "squad_v2"
raw_datasets = load_dataset(DATASET_NAME, split='train+validation')
raw_datasets = raw_datasets.filter(
    lambda x: len(x['answers']['text']) > 0
)

# Tạo mô hình và tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    # Mã hóa đầu vào văn bản
    encoded_input = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    # Di chuyển các tensor đến thiết bị (GPU hoặc CPU)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    # Lấy đầu ra của mô hình
    model_output = model(**encoded_input)
    
    # Trả về kết quả từ hàm cls_pooling
    return cls_pooling(model_output)

# Đặt tên cho cột embedding
EMBEDDING_COLUMN = 'question_embedding'

# Sử dụng hàm map để thêm cột embedding vào dataset
embeddings_dataset = raw_datasets.map(
    lambda x: {
        EMBEDDING_COLUMN: get_embeddings(
            [x['question']]
        ).detach().cpu().numpy()[0]
    }
)

# Thêm chỉ mục FAISS vào dataset với cột embedding
embeddings_dataset.add_faiss_index(column=EMBEDDING_COLUMN)

# Khởi tạo mô hình hỏi-đáp đã huấn luyện
QA_MODEL_NAME = 'NMC-29092004/distilbert-finetuned-squadv2-1'
pipe = pipeline('question-answering', model=QA_MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

# Giao diện Streamlit
st.title("Demo Hỏi-Đáp")

input_question = st.text_input("Nhập câu hỏi của bạn:", value="When did Beyonce start becoming popular?")

if input_question:
    input_quest_embedding = get_embeddings([input_question])
    input_quest_embedding = input_quest_embedding.cpu().detach().numpy()

    TOP_K = 5
    scores, samples = embeddings_dataset.get_nearest_examples(
        EMBEDDING_COLUMN, input_quest_embedding, k=TOP_K
    )

    best_answer = None
    best_score = float('inf')

    for idx, score in enumerate(scores):
        question = samples["question"][idx]
        context = samples["context"][idx]
        answer = pipe(
            question=question,
            context=context
        )
        if answer['score'] < best_score:
            best_score = answer['score']
            best_answer = answer

    # Hiển thị kết quả
    st.write("### Kết quả:")
    if best_answer:
        st.write(f"**Best Answer**: {best_answer['answer']}")
        st.write(f"**Score**: {best_answer['score']}")
        st.write(f"**Context**: {context}")
