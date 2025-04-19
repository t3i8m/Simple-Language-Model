from LMDataset import LMDataset
from torch.utils.data import DataLoader

def main():
    dataset = LMDataset("data/train.csv")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(len(dataset), dataset[0])

    return





if(__name__=="__main__"):
    main()
