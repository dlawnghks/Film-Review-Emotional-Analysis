import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.optim import AdamW
from load_data import load_data

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items() if key != "token_type_ids"}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def train_model(train_loader, model, optimizer, device, epochs=5, accumulation_steps=16):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()

        print(f"Starting epoch {epoch + 1}...")
        for step, batch in enumerate(train_loader):
            batch = {key: val.to(device) for key, val in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps

            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

if __name__ == "__main__":
    train_dir = "C:\\Users\\Administrator\\Desktop\\프로젝트\\data\\aclImdb\\train"

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    print(f"Train directory: {train_dir}")

    print("Loading data...")
    train_reviews, train_labels = load_data(train_dir)

    # 데이터 크기 축소
    train_reviews = train_reviews[:200]
    train_labels = train_labels[:200]

    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    max_length = 32  # 입력 길이 축소
    train_encodings = tokenizer(
        train_reviews, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    train_dataset = SentimentDataset(train_encodings, train_labels)
    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=data_collator
    )

    device = torch.device("cpu")  # CPU 사용
    print(f"Using device: {device}")

    model = DistilBertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)

    print("Starting training...")
    train_model(train_loader, model, optimizer, device)

    print("Saving model...")
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    print("Model and tokenizer saved to 'saved_model'")
