import torch.nn as nn

VOCAB_SIZE = 30522 # BERT tokenizer
EMB_DIM    = 64
HID_DIM    = 128
WINDOW     = 5 # context

class FeedforwardLM(nn.Module):
    def __init__(self):
        super().__init__() 
        self.embeding = nn.Embedding(VOCAB_SIZE, EMB_DIM) #projection layer
        self.fc1 = nn.Linear(EMB_DIM, HID_DIM) # fully-connected 1st layer
        self.act = nn.ReLU() # max(0,x)
        self.fc2 =  nn.Linear(HID_DIM, VOCAB_SIZE) # output layer

    def forward(self, x): # x shape: (batch, WINDOW)
        e = self.embeding(x) # (batch, WINDOW, EMB_DIM)
        flat = e.mean(dim=1) #  (batch, EMB_DIM)
        h = self.act(self.fc1(flat)) # (batch, HID_DIM)
        logits = self.fc2(h)    # (batch, VOCAB_SIZE) raw output with no softmax
        return logits
