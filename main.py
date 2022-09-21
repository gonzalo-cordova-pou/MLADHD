import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

from datasetBegin import ScreemsetDataset
from NeuralNetworkBegin import NeuralNetwork

def main():
    training_data = ScreemsetDataset("data/labels.csv","data")
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

main()