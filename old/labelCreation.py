import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    labels = open("data/labels.csv", "w")
    all = os.listdir("data")
    class_id = 0
    for i in range(len(all)):
        if not '.' in all[i]:
            files = os.listdir("data/"+all[i])
            for f in files:
                labels.write(all[i]+"/"+f+ ", "+str(class_id)+ "\n")
            class_id += 1
        
main()
