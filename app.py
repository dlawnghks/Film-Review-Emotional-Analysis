from flask import Flask, request, render_template
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from googletrans import Translator

app = Flask(__name__)

# 모델 및 토크나이저 로드
model_path = "saved_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()

# 번역기 초기화
translator = Translator()

# 감정 분석 함수
def predict_sentiment(text):
    encodings = tokenizer(text, truncation=True, padding=True, max_length=32, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in encodings.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        sentiment = "긍정적" if prediction == 1 else "부정적"
    return sentiment

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        action = request.form.get("action", "")
        user_input = request.form.get("review", "")

        if action == "translate":
            # 번역 작업
            translated_text = translator.translate(user_input, src="auto", dest="en").text
            return render_template("index.html", user_input=user_input, translated_text=translated_text)

        elif action == "analyze":
            # 감정 분석 작업
            prediction = predict_sentiment(user_input)
            return render_template("index.html", user_input=user_input, prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
