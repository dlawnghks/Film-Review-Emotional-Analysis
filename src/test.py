import os
import sys
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from preprocess import preprocess_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = "C:\\Users\\Administrator\\Desktop\\프로젝트\\saved_model"

if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found: {model_dir}")

model = DistilBertForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.to(device)

# 새로운 데이터
new_reviews = [
    "This movie was an absolute masterpiece. The plot was thrilling and the acting was phenomenal.",
    "I wasted two hours watching this film. It was boring and had no story."
]

# 리뷰 전처리
new_encodings = preprocess_data(new_reviews, tokenizer=tokenizer, max_length=64)

# 입력 데이터로 GPU 또는 CPU로 이동
inputs = {key: val.to(device) for key, val in new_encodings.items() if key != "token_type_ids"}

# 모델로 예측
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

# 결과 출력
for review, pred in zip(new_reviews, predictions):
    sentiment = "Positive" if pred.item() == 1 else "Negative"
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")
