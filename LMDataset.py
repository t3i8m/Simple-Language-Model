from torch.utils.data import Dataset
import torch
WINDOW = 5

class LMDataset(Dataset):
     
    def __init__(self, csv_path):
        # "data/train.csv"
        train_data = self._load_X_Y(csv_path)
        self.X = train_data[0]
        self.y = train_data[1]

        self.X_tensor = torch.tensor(self.X, dtype=torch.long)   # (N, window)
        self.y_tensor = torch.tensor(self.y,  dtype=torch.long) # (N, )

    def get_X(self)->(torch.Tensor):
        return self.X_tensor
    
    def get_y(self)->(torch.Tensor):
        return self.y_tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    @staticmethod
    def _load_X_Y(csv_path)->(tuple):
        X = []
        Y = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            headeer = f.readline()
            for line in f:
                parts = line.strip().split(",")
                ids = [int(n) for n in parts]     
                X.append(ids[:WINDOW])           
                Y.append(ids[WINDOW])  
        return (X,Y)
