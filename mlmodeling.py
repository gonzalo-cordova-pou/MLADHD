import torch
import json
import numpy as np
from datetime import datetime as dt
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import sampler, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import wandb
import random
import os
import glob

idx_to_class = {0: 'not_work', 1: 'work'}

class MLADHD():
    """
    This class is a wrapper for the ML models. It will be used to train and test the models.
    It will also save the models and the results.
    """

    def __init__(self, name, data_dir, models_dir, hyperparams, date=None, wandb=False):
        """
        :param name: Name of the model
        :param data_dir: Directory where the data is stored
        :param models_dir: Directory where the models will be stored
        :param hyperparams: Dictionary with the hyperparameters
        :param date: Date of the model. If None, it will be the current date
        :param wandb: If True, it will use wandb to log the results
        """
        
        self.name = name
        if date is None:
            self.date = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.date = date
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.hyperparams = hyperparams
        self.wandb = wandb
        
        self.running_loss = None
        self.criterio = None
        self.model = None
        self.train = None
        self.test = None
        self.loss = None
        self.accuracy = None
        self.classes = None
    
    def set_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams

    def load_split_dataset(self, valid_size = .2):
        """
        This function will load the dataset and split it into train and test
        :param valid_size: Percentage of the dataset that will be used for validation
        :return: None
        """
        
        # TBD: What kind of transforms can help solve the problem?
        try:
            if self.hyperparams['train_transforms'] == 'default':
                train_transf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),])
            else:
                print("Choose a valid train_transforms: default")
                return None
            
            if self.hyperparams['test_transforms'] == 'default':
                test_transf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
            else:
                print("Choose a valid test_transforms: default")
                return None
        except:
            raise Exception("You need to set the hyperparams first")
        
        #Creating the datasets
        #The number of classes will be the number of dirs
        train_data = datasets.ImageFolder(self.data_dir, transform=train_transf)
        test_data = datasets.ImageFolder(self.data_dir, transform=test_transf)
        self.classes = len(train_data.classes)
        n_train = len(train_data)
        indices = list(range(n_train))
        split = int(np.floor(valid_size * n_train))
        np.random.shuffle(indices)
        train_idx, test_idx = indices[split:], indices[:split]
        
        train_sampler =  sampler.SubsetRandomSampler(train_idx)
        test_sampler = sampler.SubsetRandomSampler(test_idx)
        
        # batch_size is the number of images that will be processed at the same time
        trainloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.hyperparams['batch_size'])
        testloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.hyperparams['batch_size'])
        self.train = trainloader
        self.test = testloader
        print('Train size: ', len(trainloader))
        print('Test size: ', len(testloader))
        print("Trainloader and Testloader created. Access them with self.train and self.test")

    def create_model(self): 
        """
        This function will create the model from a pretrained model and add
        a new classifier to it to fit the problem at hand (number of classes)
        """

        if self.hyperparams is None:
            print("You need to set the hyperparams first")
            return None
        if self.train is None:
            print("You need to load the dataset first")
            return None
        
        #define the model
        if self.hyperparams['pretrained_model'] == 'resnet50':
            # import the pretrained model
            from torchvision.models import resnet50, ResNet50_Weights
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif self.hyperparams['pretrained_model'] == 'vgg16':
            from torchvision.models import vgg16, VGG16_Weights
            self.model = vgg16(weights=VGG16_Weights.DEFAULT)
        else:
            print("Choose a valid pretrained_model: resnet50, vgg16")
            return None
        
        if self.hyperparams['freeze_pretrained_model']:
            for p in self.model.parameters():
                p.requires_grad = False
        
        fc_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(fc_inputs, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, self.classes),
                                        nn.LogSoftmax(dim=1))
        
        # define the loss function
        if self.hyperparams['loss'] == 'NLLLoss':
            self.criterio = nn.NLLLoss()
        elif self.hyperparams['loss'] == 'CrossEntropyLoss':
            self.criterio = nn.CrossEntropyLoss()
        else:
            print("Choose a valid loss: NLLLoss, CrossEntropyLoss")
            return None
    
    def train_model(self, save_model=True):
        """
        This function will train the model
        :param save_model: If True, it will save the model after training it
        :return: None
        """

        #define GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # define the optimizer
        if self.hyperparams['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])
        elif self.hyperparams['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.hyperparams['lr'])
        else:
            print("Choose a valid optimizer: Adam, SGD")
            return None

        self.running_loss = []
        for e in range(self.hyperparams['epochs']):
            running_loss = []
            for inputs, labels in self.train:
                #send to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = self.model.forward(inputs)
                loss = self.criterio(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item()*100/len(self.train))
            print("Epoch: {}/{}.. ".format(e+1, self.hyperparams['epochs']),
                    "Training Loss: {:.3f}.. ".format(running_loss[-1]))
            if self.wandb:
                wandb.log({"train_loss": running_loss[-1]})
            self.running_loss.append(running_loss)
        
        print("Training finished!")

        if save_model:
            torch.save(self.model,self.models_dir+self.name+'_'+self.hyperparams['pretrained_model']+'_'+self.date+'.pth')
            # save the hyperparams in a json file
            with open(self.models_dir+self.name+'_'+self.hyperparams['pretrained_model']+'_'+self.date+'.json', 'w') as fp:
                json.dump(self.hyperparams, fp)
            print("Model saved as: ", self.name+'_'+self.hyperparams['pretrained_model']+'_'+self.date+'.pth')
    
    def predict(self, image_path):
        """
        This function will predict the class of an image given its path
        :param image_path: path to the image
        :return: the real class and the predicted class (and the probability)
        """
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)
        with torch.no_grad():
            self.model.eval() # set model to evaluation mode
            output = self.model(image)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            # probability
        return image_path.split('/')[-2], idx_to_class[top_class.cpu().numpy()[0][0]], round(top_p.cpu().numpy()[0][0], 2)
    
    def test_random_images(self, data_dir, n_images=3):
        """
        This function will test the model with random images from a directory
        :param data_dir: directory with the images
        :param n_images: number of images to test
        :return: None
        """

        # create a plot for the images
        fig, axs = plt.subplots(n_images, 1, figsize=(30,30))
        for i in range(n_images):
            # get a random image
            image_path = random.choice(glob.glob(data_dir+'/*/*'))
            image = Image.open(image_path)
            # predict the image
            label, pred, prob = self.predict(image_path)
            # add the image to the plot
            axs[i].imshow(image)
            # set the title of the plot
            # prediction is correct
            if label == pred:
                axs[i].set_title('Label: '+label+' - Prediction: '+pred+' - Probability: '+str(prob), color='green')
            # prediction is wrong
            else:
                axs[i].set_title('Label: '+label+' - Prediction: '+pred+' - Probability: '+str(prob), color='red')
            # add image filename to the subplot x axis
            axs[i].set_xlabel(image_path.split('/')[-1])
            # remove the y axis
            axs[i].set_yticklabels([])
            axs[i].set_yticks([])
        plt.show()

    def test_model(self):
        """
        This function will test the model with the test dataset and return the loss and accuracy 
        :return: loss and accuracy
        """
        #define GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        test_loss = 0
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = self.model.forward(inputs)
                batch_loss = self.criterio(logps, labels)
                test_loss += batch_loss.item()
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        loss = test_loss/len(self.test)
        accuracy = accuracy/len(self.test)
        return loss, accuracy

    def load_model(self, model_path):
        """
        This function will load a model from a path and save it in the model attribute
        :param model_path: path to the model
        :return: None
        """
        self.model = torch.load(model_path)
        print("Model loaded from: ", model_path)

def main():
    """
    Example of how to use the functions in this module
    CREATING A NEW MODEL FROM SCRATCH (with a pytorch pretrained model)
    """
    
    model_name = 'projectmodel'
    data_dir = './data/'
    models_dir = './models/'
    valid_size = 0.2
    
    hyperparams = {
        'lr': 0.003, 
        'epochs': 1, 
        'batch_size': 64,
        'optimizer': 'Adam',            # options: Adam, SGD
        'loss': 'NLLLoss',              # options: NLLLoss, CrossEntropyLoss
        'pretrained_model': 'resnet50', # options: resnet50, vgg16
        'freeze_pretrained_model': True,# options: True, False
        'train_transforms': 'default',  # options: default
        'test_transforms': 'default'    # options: default
    }

    MLADHD = MLADHD(model_name, data_dir, models_dir, hyperparams)
    MLADHD.load_split_dataset(valid_size)
    MLADHD.create_model()
    MLADHD.train_model()
    loss, accuracy = MLADHD.test_model()
    print(loss, accuracy)

def main2():
    """
    Example of how to use the functions in this module loading
    FROM OUR PRETRAINED MODEL
    """

    data_dir = './data/'
    models_dir = './models/'
    pretrained_model = 'projectmodel_resnet50_2020-05-05_16-00-00.pth'
    hyperparams = 'projectmodel_resnet50_2020-05-05_16-00-00.json'
    
    model_name = pretrained_model.split('_')[0]

    print("Loading hyperparams...")
    with open(hyperparams, 'r') as fp:
        hyperparams = json.load(fp)
    
    # when loading a pretrained model, the date is updated to the current date
    # so that the model is not overwritten (see __init__)
    MLADHD = MLADHD(model_name, data_dir, models_dir, hyperparams)

    print("Loading model...")
    MLADHD.model = torch.load(models_dir+pretrained_model)

    MLADHD.load_split_dataset()
    loss, accuracy = MLADHD.test_model()
    print(loss, accuracy)

'''
Note: I remove this because I import this file in the DEMO.ipynb notebook and I
execute the main function in the notebook

def main():
    data_dir = 'data/'
    #define datasets and the percent of train and test
    train, test = load_split_dataset(data_dir, .3)
    model,loss = create_model(train)
    test_loss,accuracy = test_model(model,test)

    print(test_loss,accuracy)
    plt.plot(loss,label='Training loss')
    plt.legend(frameon=False)
    plt.show()

main()
'''
