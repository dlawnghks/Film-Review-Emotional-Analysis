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

    train_dir = "data/aclImdb/train"
    train_reviews, train_labels = load_data(train_dir)
    train_encodings = preprocess_data(train_reviews)


    train_dataset = SentimentDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_model(train_loader, model, optimizer, device)
