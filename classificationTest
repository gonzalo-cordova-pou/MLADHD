import cv2
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.autograd import Variable


def predict_image(model,image):
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_t = test_transforms(image).float()
    img_t = img_t.unsqueeze_(0)
    input = Variable(img_t)
    input = input.to(device)
    output = model(input)
    classe = output.data.cpu().numpy().argmax()
    return classe

def main(model_name,image):
    model=torch.load(model_name)
    model.eval()
    to_pil = transforms.ToPILImage()
    classes = ["game","work"]

    a = cv2.imread(image)
    t = transforms.functional.to_tensor(a)
    t = to_pil(t)
    classe = predict_image(model,t)
    print(classes[classe])
    fig=plt.figure(figsize=(1,1))
    
    plt.imshow(image)

main('projectmodel.pth',"a (1).jpg")