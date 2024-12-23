import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "saved_model_multilingual"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()

print("Enter a review to predict sentiment (type 'exit' to quit):")
while True:
    review = input("Your review: ")
    if review.lower() == "exit":
        print("Exiting sentiment analysis...")
        break
    
    encodings = tokenizer(review, truncation=True, padding=True, max_length=32, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in encodings.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        sentiment = "긍정적" if prediction == 1 else "부정적"
        print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")
