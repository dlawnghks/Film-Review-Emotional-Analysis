import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    AdamW,
    get_scheduler,
)
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
from sklearn.utils.class_weight import compute_class_weight
import random

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

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    # 한국어 데이터 경로 설정
    korean_data_dir = "data/aclImdb_k/train"

    print("Loading Korean data...")
    korean_reviews, korean_labels = load_balanced_data(korean_data_dir, sample_size_per_class=1000)

    # 데이터 준비
    combined_reviews = korean_reviews
    combined_labels = korean_labels

    # 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=combined_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 토크나이저 로드 및 인코딩
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
    encodings = tokenizer(
        combined_reviews, truncation=True, padding=True, max_length=64, return_tensors="pt"
    )
    dataset = SentimentDataset(encodings, combined_labels)

    # 데이터 분할 (90% Train, 10% Validation)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader 정의
    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=data_collator, num_workers=0, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-small", num_labels=2)
    model.gradient_checkpointing_enable()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 10
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)

    loss_fn = CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler(init_scale=2.0**1)

    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 3

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {key: val.to(device) for key, val in batch.items()}
            with autocast():
                outputs = model(**batch)
                logits = outputs.logits.float()
                loss = loss_fn(logits, batch["labels"]) / 4  # Gradient Accumulation Steps

            scaler.scale(loss).backward()

            if (step + 1) % 4 == 0 or (step + 1) == len(train_loader):  # Update every 4 steps
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * 4

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits.float()
                loss = loss_fn(logits, batch["labels"])
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Validation loss improved. Saving model...")
            model.save_pretrained("saved_model_korean")
            tokenizer.save_pretrained("saved_model_korean")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    print("Training completed.")
