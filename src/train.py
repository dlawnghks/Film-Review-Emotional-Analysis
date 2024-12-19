import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    AdamW,
    get_scheduler,
)
from torch.nn import CrossEntropyLoss
import random

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# 데이터셋 정의
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

# 학습 함수
def train_model(train_loader, model, optimizer, device, scheduler, loss_fn, epochs=10, accumulation_steps=16):
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()
    model.train()
    early_stopping = EarlyStopping(patience=3)

    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()

        print(f"Starting epoch {epoch + 1}...")
        for step, batch in enumerate(train_loader):
            batch = {key: val.to(device) for key, val in batch.items()}
            
            with autocast():
                outputs = model(**batch)
                logits = outputs.logits
                loss = loss_fn(logits, batch["labels"]) / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            # Scheduler step
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        # Early stopping check
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

# 데이터 로드 함수 (균형 유지)
def load_balanced_data(data_dir, sample_size_per_class=None):
    reviews, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        files = os.listdir(dir_name)
        if sample_size_per_class:
            files = random.sample(files, sample_size_per_class)
        for fname in files:
            with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return reviews, labels

# 메인 함수
if __name__ == "__main__":
    # 데이터 경로
    train_dir = "C:\\Users\\Administrator\\Desktop\\프로젝트\\data\\aclImdb\\train"

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    print(f"Train directory: {train_dir}")

    print("Loading data...")
    train_reviews, train_labels = load_balanced_data(train_dir, sample_size_per_class=500)

    # 클래스 비율 확인
    print(f"Positive samples: {sum(train_labels)}")
    print(f"Negative samples: {len(train_labels) - sum(train_labels)}")

    if sum(train_labels) == 0 or (len(train_labels) - sum(train_labels)) == 0:
        raise ValueError("Data imbalance detected: Ensure both positive and negative samples are present.")

    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    max_length = 64  # 최대 길이
    train_encodings = tokenizer(
        train_reviews, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    train_dataset = SentimentDataset(train_encodings, train_labels)
    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Define epochs and training steps
    epochs = 10  # Number of epochs
    num_training_steps = epochs * len(train_loader)

    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=10, num_training_steps=num_training_steps)

    # Define loss function
    loss_fn = CrossEntropyLoss()

    print("Starting training...")
    train_model(train_loader, model, optimizer, device, scheduler, loss_fn, epochs=epochs)

    print("Saving model...")
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    print("Model and tokenizer saved to 'saved_model'")
