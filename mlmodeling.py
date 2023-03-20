import torch
import json
import numpy as np
from datetime import datetime as dt
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import sampler, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import wandb
import random
import os
import glob
import time

# Binary classification problem
idx_to_class = {0: 'focused', 1: 'distracted'}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

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

        self.trainloader = None
        self.validloader = None
        self.testloader = None
        
        self.training_history = {
            'train_loss': [],
            'valid_loss': [],
            'train_acc': [],
            'valid_acc': []
        }
        self.test_loss = None
        self.test_acc = None
        self.test_precision = None
        self.test_recall = None
        self.test_f1 = None

        self.criterion = None
        self.model = None
        self.classes = None
    
    def set_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams

    def load_split_dataset(self, percent=(0.8, 0.1, 0.1)):
        """
        This function will load the dataset and split it into train, valid and test
        :param percent: Tuple with the percentage of train, valid and test
        """
        
        try:
            if self.hyperparams['train_transforms'] == 'default':
                train_transf = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                ])
            else:
                print("Choose a valid train_transforms: default")
                return None
            
            if self.hyperparams['valid_transforms'] == 'default':
                valid_transf = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                ])
            else:
                print("Choose a valid valid_transforms: default")
                return None
            
            if self.hyperparams['test_transforms'] == 'default':
                test_transf = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                ])
            else:
                print("Choose a valid test_transforms: default")
                return None
            
        except:
            raise Exception("You need to set the hyperparams first")
        
        #Creating the datasets
        #The number of classes will be the number of dirs
        data = datasets.ImageFolder(self.data_dir, transform=train_transf)
        self.classes = len(data.classes)

        #Creating the train, valid and test sets using random_split
        train_size = int(percent[0] * len(data))
        valid_size = int(percent[1] * len(data))
        test_size = len(data) - train_size - valid_size
        train_data, valid_data, test_data = torch.utils.data.random_split(data, [train_size, valid_size, test_size])

        #Creating the dataloaders
        self.trainloader = DataLoader(train_data, batch_size=self.hyperparams['batch_size'], shuffle=True)
        self.validloader = DataLoader(valid_data, batch_size=self.hyperparams['batch_size'], shuffle=True)
        self.testloader = DataLoader(test_data, batch_size=self.hyperparams['batch_size'], shuffle=True)

        print("Train size: ", len(train_data))
        print("Valid size: ", len(valid_data))
        print("Test size: ", len(test_data))

    def create_model(self): 
        """
        This function will create the model from a pretrained model and add
        a new classifier to it to fit the problem at hand (number of classes)
        """

        if self.hyperparams is None:
            print("You need to set the hyperparams first")
            return None
        if self.trainloader is None:
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
            self.criterion = nn.NLLLoss()
        elif self.hyperparams['loss'] == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
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
        self.model.to(device)

        # define the optimizer
        if self.hyperparams['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])
        elif self.hyperparams['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.hyperparams['lr'])
        else:
            print("Choose a valid optimizer: Adam, SGD")
            return None

        for epoch in range(self.hyperparams['epochs']):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(epoch+1, self.hyperparams['epochs']))
            # Loss and Accuracy within the epoch
            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0
            self.model.train()
            for i, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                # Clean existing gradients
                optimizer.zero_grad()
                # Forward pass - compute outputs on input data using the model
                outputs = self.model(inputs)
                # Compute loss
                loss = self.criterion(outputs, labels)
                # Backpropagate the gradients
                loss.backward()
                # Update the parameters
                optimizer.step()
                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)
                # Compute the accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)
                if i % 5 == 0:
                    print("Batch number: {:03d}/{:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, len(self.trainloader), loss.item(), acc.item()))
                if self.wandb:
                    wandb.log({"Train Loss": loss.item(), "Train Accuracy": acc.item()})
        
            # Validation - No gradient tracking needed
            with torch.no_grad():
                # Set to evaluation mode
                self.model.eval()
                # Validation loop
                for j, (inputs, labels) in enumerate(self.validloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Forward pass - compute outputs on input data using the model
                    outputs = self.model(inputs)
                    # Compute loss
                    loss = self.criterion(outputs, labels)
                    # Compute the total loss for the batch and add it to test_loss
                    valid_loss += loss.item() * inputs.size(0)
                    # Calculate accuracy
                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))
                    # Convert correct_counts to float and then compute the mean
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))
                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)
                    print("Valid. Batch number: {:03d}/{:03d}, Valid: Loss: {:.4f}, Accuracy: {:.4f}".format(j, len(self.validloader), loss.item(), acc.item()))
                    if self.wandb:
                        wandb.log({"Valid Loss": loss.item(), "Valid Accuracy": acc.item()})
            # Compute the average losses and accuracy (for both training and validation) for the epoch
            avg_train_loss = train_loss/float(len(self.trainloader.dataset))
            avg_train_acc = train_acc/float(len(self.trainloader.dataset))
            avg_valid_loss = valid_loss/float(len(self.validloader.dataset))
            avg_valid_acc = valid_acc/float(len(self.validloader.dataset))
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_acc'].append(avg_train_acc)
            self.training_history['valid_loss'].append(avg_valid_loss)
            self.training_history['valid_acc'].append(avg_valid_acc)
            epoch_end = time.time()
            print("_"*10)
            print("Epoch : {:03d}\nTraining: Loss: {:.4f}, Accuracy: {:.4f}\nValidation: Loss: {:.4f}, Accuracy: {:.4f}".format(epoch+1, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc))
            print("_"*10)
            if self.wandb:
                wandb.log({"Epoch": epoch+1, "Train Loss": avg_train_loss, "Train Accuracy": avg_train_acc, "Valid Loss": avg_valid_loss, "Valid Accuracy": avg_valid_acc})

        if save_model:
            torch.save(self.model,self.models_dir+self.name+'_'+self.hyperparams['pretrained_model']+'_'+self.date+'.pth')
            # save the hyperparams in a json file
            with open(self.models_dir+self.name+'_'+self.hyperparams['pretrained_model']+'_'+self.date+'.json', 'w') as fp:
                json.dump(self.hyperparams, fp)
            print("Model saved as: ", self.name+'_'+self.hyperparams['pretrained_model']+'_'+self.date+'.pth')

    def plot_training(self):
        """
        This function will plot the training and validation loss and accuracy
        :return: None
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Training Loss')
        plt.plot(self.training_history['valid_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss')
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['train_acc'], label='Training Accuracy')
        plt.plot(self.training_history['valid_acc'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

    def test_model(self):
        """
        This function will test the model with the following metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - Confusion Matrix
        :return: None
        """
        test_loss = 0.0
        test_acc = 0.0
        test_precision = 0.0
        test_recall = 0.0
        test_f1 = 0.0
        y_true = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(self.testloader):
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass - compute outputs on input data using the model
                outputs = self.model(inputs)
                # Compute loss
                loss = self.criterion(outputs, labels)
                # Compute the total loss for the batch and add it to test_loss
                test_loss += loss.item()*inputs.size(0)
                # Calculate accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                # Compute total accuracy in the whole batch and add to test_acc
                test_acc += acc.item()*inputs.size(0)
                # Compute precision, recall and f1 score
                precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted')
                test_precision += precision*inputs.size(0)
                test_recall += recall*inputs.size(0)
                test_f1 += f1*inputs.size(0)
                # Add true and predicted labels for the confusion matrix
                y_true += labels.cpu().numpy().tolist()
                y_pred += predictions.cpu().numpy().tolist()
                print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(j, loss.item(), acc.item(), precision, recall, f1))
        # Compute the average losses, accuracy, precision, recall and f1 score
        self.test_loss = test_loss/len(self.testloader.dataset)
        self.test_acc = test_acc/float(len(self.testloader.dataset))
        self.test_precision = test_precision/float(len(self.testloader.dataset))
        self.test_recall = test_recall/float(len(self.testloader.dataset))
        self.test_f1 = test_f1/float(len(self.testloader.dataset))
        print("Test: Loss: {:.4f}, Accuracy: {:.4f}%, Precision: {:.4f}%, Recall: {:.4f}%, F1 Score: {:.4f}%".format(self.test_loss, self.test_acc*100, self.test_precision*100, self.test_recall*100, self.test_f1*100))
        if self.wandb:
            wandb.log({"Test Loss": self.test_loss, "Test Accuracy": self.test_acc, "Test Precision": self.test_precision, "Test Recall": self.test_recall, "Test F1 Score": self.test_f1})
        # Plot the confusion matrix
        plot_confusion_matrix(y_true, y_pred)

    def predict(self, image_path):
        """
        This function will predict the class of an image
        :param image_path: The path of the image
        :return: Tuple (real class, predicted class, probability)
        """
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            self.model.eval()
            output = self.model(image)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
        return image_path.split('/')[-2].split("_")[-1], idx_to_class[top_class.cpu().numpy()[0][0]], round(top_p.cpu().numpy()[0][0], 2)

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

    def load_model(self, model_path):
        """
        This function will load a model from a path and save it in the model attribute
        :param model_path: path to the model
        :return: None
        """
        self.model = torch.load(model_path)
        print("Model loaded from: ", model_path)

def plot_confusion_matrix(y_true, y_pred, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="g", cmap=cmap)
    plt.title("Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()

def main():
    """
    Example of how to use the functions in this module
    CREATING A NEW MODEL FROM SCRATCH (with a pytorch pretrained model)
    """
    
    model_name = 'projectmodel'
    data_dir = './data/'
    models_dir = './models/'
    
    hyperparams = {
        'lr': 0.003, 
        'epochs': 1, 
        'batch_size': 64,
        'optimizer': 'Adam',            # options: Adam, SGD
        'loss': 'NLLLoss',              # options: NLLLoss, CrossEntropyLoss
        'pretrained_model': 'resnet50', # options: resnet50, vgg16
        'freeze_pretrained_model': True,# options: True, False
        'train_transforms': 'default',  # options: default
        'valid_transforms': 'default',  # options: default
        'test_transforms': 'default'    # options: default
    }

    MLADHD = MLADHD(model_name, data_dir, models_dir, hyperparams)
    MLADHD.load_split_dataset((0.8, 0.1, 0.1))
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