import gzip
import json
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class SentimentDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        review = item['reviewText']
        emotion = item['emotion']
        
        inputs = self.tokenizer.encode_plus(
            f"sentiment: {review}",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        targets = self.tokenizer.encode_plus(
            emotion,
            max_length=10,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }

def train_model(model, train_dataloader, val_dataloader, num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} - Validation loss: {avg_val_loss:.4f}")
    
    return model

def main():
    tokenizer = MT5Tokenizer.from_pretrained("mgmahi/mlt5-multilingual-sentiment")
    model = MT5ForConditionalGeneration.from_pretrained("mgmahi/mlt5-multilingual-sentiment")

    dataset = SentimentDataset('data/labeled_Electronics_5.json.gz', tokenizer)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    trained_model = train_model(model, train_dataloader, val_dataloader)

    trained_model.save_pretrained("models/fine_tuned_mT5_sentiment")
    tokenizer.save_pretrained("models/fine_tuned_mT5_sentiment")

if __name__ == "__main__":
    main()