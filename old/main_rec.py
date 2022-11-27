import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import cv2

from torch.autograd import Variable

import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import torchvision.transforms.functional as TF


data_dir = 'data'
test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('aerialmodel.pth')
model.eval()

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index
"""
def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels, classes
"""
to_pil = transforms.ToPILImage()
"""
images, labels, classes = get_random_images(5)
fig=plt.figure(figsize=(10,10))

for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()"""

classes = ["game","work"]

a = cv2.imread("b (3).jpg")
t = TF.to_tensor(a)
#t.unsqueeze_(0)
t = to_pil(t)
index = predict_image(t)
print(classes[index])
