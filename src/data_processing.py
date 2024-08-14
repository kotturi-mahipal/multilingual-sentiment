import gzip
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import multiprocessing as mp

class AmazonDataset(Dataset):
    def __init__(self, file_path, cache_size=1000):
        self.file_path = file_path
        self.cache = []
        self.cache_size = cache_size
        self.load_cache()

    def load_cache(self):
        with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
            for _ in range(self.cache_size):
                try:
                    line = next(f)
                    self.cache.append(json.loads(line.strip()))
                except StopIteration:
                    break

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        # Try to get 'reviewText', if not present, use 'review' or the first string value
        item = self.cache[idx]
        if 'reviewText' in item:
            return item['reviewText']
        elif 'review' in item:
            return item['review']
        else:
            # Return the first string value found
            for value in item.values():
                if isinstance(value, str):
                    return value
        raise KeyError(f"No suitable text field found in item: {item}")

def process_batch(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("michellejieli/emotion_text_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("michellejieli/emotion_text_classifier").to(device)

    emotions = []
    with torch.no_grad():
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        emotions.extend(model.config.id2label[pred.item()] for pred in predictions)
    return emotions

def label_emotions(dataset, batch_size=32, num_workers=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    with mp.Pool(num_workers) as pool:
        emotions = pool.map(process_batch, [batch for batch in dataloader])

    emotions = [item for batch in emotions for item in batch]
    return emotions

def process_and_label_data():
    input_file_path = "data/Electronics_5.json.gz"
    output_file_path = "data/labeled_Electronics_5.json.gz"
    
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    dataset = AmazonDataset(input_file_path)
    emotions = label_emotions(dataset)
    
    with gzip.open(output_file_path, 'wt', encoding='utf-8') as f:
        for review, emotion in zip(dataset.cache, emotions):
            review['emotion'] = emotion
            f.write(json.dumps(review) + '\n')
    
    print(f"Processed and labeled data saved to: {output_file_path}")


if __name__ == "__main__":
    process_and_label_data()
