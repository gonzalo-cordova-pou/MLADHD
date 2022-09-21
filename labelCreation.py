import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import isfile, join

def main():
    all = os.listdir("data")
    labels = open("data/labels.csv", "w")
    #dirs = []
    #print(all)
    #for a in all:
    #    if not isfile(a):
    #        dirs+=a
    #print(dirs)
    for i in range(len(all)):
        if all[i] != "data/labels.csv":
            files = os.listdir("data/"+all[i])
            for f in files:
                labels.write(f+ ", "+str(i)+ "\n")
        
main()


