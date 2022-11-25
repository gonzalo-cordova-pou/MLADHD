import torch
import numpy as np

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import sampler, DataLoader
from torchvision.models import resnet50, ResNet50_Weights

import matplotlib.pyplot as plt

def load_split_dataset(data_dir, valid_size = .2):
    #What kind of transforms can help solve the problem?
    train_transf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),])
    test_transf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
   
    #Creating the datasets
    #The number of classes will be the number of dirs
    train_data = datasets.ImageFolder(data_dir, transform=train_transf)
    test_data = datasets.ImageFolder(data_dir, transform=test_transf)
    n_train = len(train_data)
    indices = list(range(n_train))
    split = int(np.floor(valid_size * n_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler =  sampler.SubsetRandomSampler(train_idx)
    test_sampler = sampler.SubsetRandomSampler(test_idx)
    #batch_size is how many images will be analysed in each iteration, minimize noise
    trainloader = DataLoader(train_data, sampler=train_sampler, batch_size=64)
    testloader = DataLoader(test_data, sampler=test_sampler, batch_size=64)
    return trainloader, testloader


def create_model():
    data_dir = 'data/'
    #define datasets and the percent of train and test
    train, test = load_split_dataset(data_dir, .2)
    print(train.dataset.classes)

    #define the model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 10),
                                    nn.LogSoftmax(dim=1))
    criterio = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    
    #define GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    running_loss = []
    #for epoch in range(epochs):
    for inputs, labels in train:
            #send to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterio(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item()*100/len(train))
    torch.save(model, 'projectmodel.pth')
    return running_loss

def main():
    loss = create_model(); 
    plt.plot(label='Training loss')
    plt.legend(frameon=False)
    plt.show()

main()