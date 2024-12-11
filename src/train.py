import torch
from transformers import BertForSequenceClassification  # AdamW 임포트 제거
from torch.optim import AdamW  # PyTorch의 AdamW 사용
from torch.utils.data import DataLoader, Dataset
from preprocess import preprocess_data
from load_data import load_data

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 입력 데이터 처리
        inputs = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        # 라벨 데이터 처리 (정수형이므로 clone().detach() 제거)
        label = torch.tensor(self.labels[idx])
        
        return inputs, label

def train_model(train_loader, model, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            # 입력 데이터를 디바이스로 이동
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

if __name__ == "__main__":
    # 데이터 로드 및 전처리
    train_dir = "data/aclImdb/train"
    train_reviews, train_labels = load_data(train_dir)
    train_encodings = preprocess_data(train_reviews)

    # 데이터셋 및 데이터로더 생성
    train_dataset = SentimentDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 초기화 (classifier 가중치는 새로 초기화됨)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    # 옵티마이저 초기화 (torch.optim.AdamW 사용)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 모델 학습
    train_model(train_loader, model, optimizer, device)
