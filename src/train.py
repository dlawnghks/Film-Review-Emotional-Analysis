import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    AdamW,
    get_scheduler,
)
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
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

    english_data_dir = "data/aclImdb/train"
    korean_data_dir = "data/aclImdb_k/train"

    print("Loading English data...")
    english_reviews, english_labels = load_balanced_data(english_data_dir, sample_size_per_class=500)

    print("Loading Korean data...")
    korean_reviews, korean_labels = load_balanced_data(korean_data_dir, sample_size_per_class=500)

    combined_reviews = english_reviews + korean_reviews
    combined_labels = english_labels + korean_labels

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    train_encodings = tokenizer(
        combined_reviews, truncation=True, padding=True, max_length=64, return_tensors="pt"
    )
    train_dataset = SentimentDataset(train_encodings, combined_labels)
    data_collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # 배치 크기 감소
        shuffle=True,
        collate_fn=data_collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
    model.gradient_checkpointing_enable()
    model.to(device)
    torch.cuda.empty_cache()  # GPU 메모리 캐시 초기화

    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 12
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=10, num_training_steps=num_training_steps)

    loss_fn = CrossEntropyLoss()
    scaler = GradScaler(init_scale=2.0**2)  # GradScaler 초기값 조정

    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {key: val.to(device) for key, val in batch.items()}
            with autocast():
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, batch["labels"]) / 8  # Gradient Accumulation

            scaler.scale(loss).backward()

            if (step + 1) % 8 == 0 or (step + 1) == len(train_loader):  # 8 스텝마다 업데이트
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * 8

            # GPU 메모리 캐시 정리
            if step % 50 == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    print("Saving model...")
    model.save_pretrained("saved_model_multilingual")
    tokenizer.save_pretrained("saved_model_multilingual")
    print("Model and tokenizer saved to 'saved_model_multilingual'")
