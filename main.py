from LMDataset import LMDataset
from torch.utils.data import DataLoader
from Trainer import Trainer
from FeedForwardLM import FeedforwardLM

def main():
    dataset = LMDataset("data/train.csv")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FeedforwardLM()
    trainer = Trainer(model, loader)

    trainer.train(5)
    print(len(dataset), dataset[0])
    return





if(__name__=="__main__"):
    main()
