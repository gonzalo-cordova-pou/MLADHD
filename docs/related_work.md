# Related work

## Index

- [Article 1](#article-1) CNN for task classification using computer screenshots for integration into dynamic calendar/task management systems
- [Article 2](#article-2) Guess What's on my Screen? Clustering Smartphone Screenshots with Active Learning
- [Article 3](#article-3) Understanding Screen Relationships from Screenshots of Smartphone Applications
- [Article 4](#article-4) Detecting Developers’ Task Switches and Types
- [Other references](#Other-references)

## Article 1

[Link to the article](http://cs231n.stanford.edu/reports/2015/pdfs/anand_avery_final.pdf)

**Title:** CNN for task classification using computer screenshots for integration into dynamic calendar/task management systems.<br>

**Authors:** Anand Sampat, Avery Haskell - Stanford University<br>

**Context:** This is an unpublished manuscript of 2015 that appears to be cited in 5 other published articles according to Google Scholar.<br>

**Summary:** This project aims to classify high level tasks (e.g. ‘checking emails’, ‘photo editing’, ‘social media’) a user does on his or her computer using deep CNN networks. The technique of transfer is implemented using scene-classification pretrained models which provide weights that significantly reduce training time by placing the solver in a state closer to the optimal state. It explores two datasets – one small (1900 training examples), and one larger (10,800 training images). Moreover, additional work in window detection and streaming data (time component) is carried out.<br>

**Core idea:**
1. Elaboration of a new screenshot labeled dataset using automated image search from search engine scraping and screenshots taken at regular intervals from a computer.
2. Overcoming data challenges: Data sparsity with data augmentation (moving windows); Data Diversity by avoiding too homogeneous data (e.g. OS, theme, color scheme); Data Cleaning by hand curation, duplicate removal and filters.
3. The backbone architecture of the model is the Hybrid CNN. This architecture is simplified by reducing the number of layers. The architecture was pre-trained on the Places205CNN dataset from MIT and by freezing some of the layers and training only the last 3 to 5 layers the new model is obtained.
4. Finally, work on data interpretation is also presented. As a result of these observations the papers explores two possibilities:
    - a) Window detection: using bounding boxes to capture different windows in the screen (OpenCV library: canny edge detection).
    - b) Streaming data: aiming to capture the idea that temporal context is important to classification by using more than one image (10 sec intervals) to make a prediction.
<br>

**Conclusion:**
- On a large dataset the best model performed reasonably with 64% accuracy using a CNN with weights from hybridCNN training and retraining the last conv layer and the two fully-connected layers on 10,800 images. 
- Despite not showing accurate results, the paper suggests that in order to better integrate this, data streaming is very important. Furthermore, obtaining contextual cues via segmentation of the image with bounding boxes can be used in conjunction with a contextually constrained deep network to improve task labeling.



## Article 2 
**(in progress...)**

[Link to the article](https://arxiv.org/pdf/1805.07964.pdf)

**Title:** Guess What's on my Screen? Clustering Smartphone Screenshots with Active Learning

**Authors:** Agnese Chiatti, Dolzodmaa Davaasuren, Nilam Ram, Prasenjit Mitra, Byron Reeves, Thomas Robinson - The Pennsylvania State University & Stanford University

**Context:** This paper was published in January 2019 as a collaboration between The Pennsylvania State University and Stanford University. Stanford University involved their [Screenomics Lab](https://screenomics.stanford.edu/) members for useful discussions and acknowledge the data and computational support provided for the experiments.
<br>

**Summary:**

<br>

**Core idea:**

<br>

**Conclusion:**


## Article 3
**(in progress...)**

[Link to the article](https://dl.acm.org/doi/pdf/10.1145/3490099.3511109)

**Title:** Understanding Screen Relationships from Screenshots of Smartphone Applications

**Authors:**  Shirin Feiz, Jason Wu, Xiaoyi Zhang, Amanda Swearngin, Titus Barik, Jeffrey Nichols

**Context:**

<br>

**Summary:**

<br>

**Core idea:**

<br>

**Conclusion:**

## Article 4
**(in progress...)**

[Link to the article](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9069309)

**Title:** Detecting Developers’ Task Switches and Types

**Authors:** Andre N. Meyer, Chris Satterfield, Manuela Zuger, Katja Kevic, Gail C. Murphy, Thomas Zimmermann, and Thomas Fritz

**Context:**

<br>

**Summary:**

<br>

**Core idea:**

<br>

**Conclusion:**

## Other references


1. T. Kekec, R. Emonet,E. Fromont,A. Tremeau,and C. Wolf ”**Contextually Constrained Deep Networks for Scene Labeling**”. [http://www.bmva.org/ bmvc/2014/files/paper032.pdf]
2. B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva. ”Learning Deep **Features for Scene Recognition** using Places Database.” Advances in Neural Information Processing Systems 27 (NIPS), 2014 [http: //places.csail.mit.edu/]
3. A. Krizhevsky,I. Sutskevar, G. Hinton. **ImageNet Classification with Deep Convolutional Neural Networks**. Advances in Neural Information Processing Systems 25 (NIPS) 2012 [http://papers.nips.cc/paper/ 4824-imagenet-classification-with-deep-convolutpdf]
4. J. Long,E. Shelhamer, and T. Darell. **Fully Convolutional Networks for Semantic Segmentation**.[http: //arxiv.org/pdf/1411.4038v1.pdf]
5. **Canny edge detection** (OpenCV) http://opencv-python-tutroals. readthedocs.org/en/latest/py_ tutorials/py_imgproc/py_canny/py_ canny.html
6. M. D. Zeiler and R. Fergus. ”Visualizing and Understanding Convolutional Networks. [http:// arxiv.org/pdf/1311.2901v3.pdf]
