from FeedForwardLM import FeedforwardLM
from torch import nn
import torch
import os
MODEL_DIR = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)

class Trainer():
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train(self, epochs):
        NUM_EPOCHS = epochs

        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for X_batch, y_batch in self.loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)          # (batch, VOCAB_SIZE)
                loss = self.loss_fn(logits, y_batch)  # y_batch shape (batch,)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"epoch {epoch+1}  |  loss {total_loss/len(self.loader):.4f}")

        save_path = os.path.join(MODEL_DIR, "fflm.pt")
        torch.save(self.model.state_dict(), save_path)
        print(f"Weights saved to {save_path}")