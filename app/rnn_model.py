import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEQ_LEN = 30


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - SEQ_LEN

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx + SEQ_LEN]),
            torch.tensor(self.data[idx + 1:idx + SEQ_LEN + 1])
        )


class LSTMModel(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


class RNNTextGenerator:
    def __init__(self, corpus, device=None, epochs=15):
        self.device = device or (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        text = " ".join(corpus)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = text.lower()
        tokens = text.split()

        counter = Counter(tokens)
        self.vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(9998))}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}

        encoded = [self.vocab.get(word, self.vocab["<UNK>"]) for word in tokens]

        self.model = LSTMModel(vocab_size=10000).to(self.device)

        if len(encoded) > SEQ_LEN:
            self.train(encoded, epochs=epochs)

    def train(self, encoded, epochs=15):
        train_dataset = TextDataset(encoded)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        self.model.train()
        for _ in range(epochs):
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs, _ = self.model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def generate_text(self, seed_text, length=50, temperature=1.0):
        self.model.eval()
        words = seed_text.lower().split()
        input_ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        hidden = None

        with torch.no_grad():
            for _ in range(length):
                output, hidden = self.model(input_tensor, hidden)
                logits = output[0, -1] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                words.append(self.inv_vocab.get(next_id, "<UNK>"))

                input_ids.append(next_id)
                input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)

        return " ".join(words)