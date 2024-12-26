import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    DistilBertForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    AdamW,
    get_scheduler,
)
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
import random


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


def train_model(train_loader, val_loader, model, optimizer, device, scheduler, loss_fn, epochs=10, accumulation_steps=16):
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=3)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
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
                scheduler.step()

            total_loss += loss.item() * accumulation_steps

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, batch["labels"])
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    train_dir = "C:\\Users\\Administrator\\Desktop\\프로젝트\\data\\aclImdb\\train"

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    print("Loading data...")
    train_reviews, train_labels = load_balanced_data(train_dir, sample_size_per_class=1000)

    print(f"Positive samples: {sum(train_labels)}")
    print(f"Negative samples: {len(train_labels) - sum(train_labels)}")

    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    max_length = 64
    encodings = tokenizer(
        train_reviews, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    dataset = SentimentDataset(encodings, train_labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 10
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=10, num_training_steps=num_training_steps)

    loss_fn = CrossEntropyLoss()

    print("Starting training...")
    train_model(train_loader, val_loader, model, optimizer, device, scheduler, loss_fn, epochs=epochs)

    print("Saving model...")
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    print("Model and tokenizer saved to 'saved_model'")
