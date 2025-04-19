from torch.utils.data import Dataset
import torch
WINDOW = 5

class LMDataset(Dataset):
     
    def __init__(self, X, y):
        train_data = self.download_X_Y()
        self.X = train_data[0]
        self.y = train_data[1]

        self.X_tensor = torch.tensor(X, dtype=torch.long)   # (N, window)
        self.y_tensor = torch.tensor(y,  dtype=torch.long) # (N, )


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def download_X_Y(self)->(list):
        X = []
        Y = []
        with open("data/train.csv") as f:
            headeer = f.readline()
            for line in f:
                parts = line.strip().split(",")
                ids = [int(n) for n in parts]     
                X.append(ids[:WINDOW])           
                Y.append(ids[WINDOW])  
        return (X,Y)
