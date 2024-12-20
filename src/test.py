import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "saved_model_multilingual"

test_reviews = [
    "This was a terrible movie. I hated it.",  # Negative
    "영화가 정말 좋았어요. 감동적이었어요.",  # Positive
    "Boring and uninteresting.",  # Negative
    "최고의 영화였습니다. 정말 추천합니다!",  # Positive
]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()

for review in test_reviews:
    encodings = tokenizer(review, truncation=True, padding=True, max_length=32, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in encodings.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")
