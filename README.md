# Computer Screenshot Classification for Boosting ADHD Productivity in a VR environment

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)

This repository contains the code for the author's Bachelor's Thesis. You can find the written report [here]().
- **Title**: Computer Screenshot Classification for Boosting ADHD Productivity in a VR environment
- **Author**: Gonzalo Córdova Pou
- **Supervisor**: Silverio Martínez-Fernández, Department of Service and Information System Engineering
- **Institution**: Universitat Politècnica de Catalunya (UPC)
    - Schools: Barcelona school of Informatics, Barcelona School of Telecommunications Engineering, School of Mathematics and Statistics
- **Co-supervisors**: David C. Shepherd (Louisiana State University), Juliana Souza (Virginia Commonwealth University)

Individuals with ADHD face significant challenges in their daily lives due to difficulties with attention, hyperactivity, and impulsivity. These challenges are especially pronounced in the workplace or educational settings, where the ability to sustain attention and manage time effectively is crucial for success. Virtual reality (VR) software has emerged as a promising tool for improving productivity in individuals with ADHD. However, the effectiveness of such software depends on the identification of potential distractions and timely intervention.

The proposed computer screenshot classification approach addresses this need by providing a means for identifying and analyzing potential distractions within VR software. By integrating Convolutional Neural Networks (CNNs), Optical Character Recognition (OCR), and Natural Language Processing (NLP), the proposed approach can accurately classify screenshots and extract features, facilitating the identification of distractions and enabling timely intervention to minimize their impact on productivity.

The implications of this research are significant, as ADHD affects a substantial portion of the population and has a significant impact on productivity and quality of life. By providing a novel approach for studying, detecting, and enhancing productivity, this research has the potential to improve outcomes for individuals with ADHD and increase the efficiency and effectiveness of workplaces and educational settings. Moreover, the proposed approach holds promise for wider applicability to other productivity studies involving computer users, where the classification of screenshots and feature extraction play a crucial role in discerning behavioral patterns.

## Index

1. [Repository structure](#repository-structure)
2. [Related Work](#related-work)
3. [Data](#data)
5. [Experiment tracking](#experiment-tracking)

---

## Repository Structure


- `docs`: Documentation files.
- `.codecarbon.config`: Configuration file for the CodeCarbon tool.
- `.gitignore`: Git ignore file.
- `CNNclassifier.py`: Python module with the CNN model and the functions to train, test and predict.
- `distraction_detector_lab.py`: Distraction Detector Loop with intervention (CNN model, for testing intervention parameters).
- `distraction_detector.py`: Distraction Detector Loop with intervention (CNN model, for deployment).
- `environment.yml`: Conda environment file.
- `Experiment_colab.ipynb`: Notebook to run experiments(train, test...) for the CNN Distraction Detector in Google Colab.
- `Experiment_local.ipynb`: Notebook to run experiments (train, test...) for the CNN Distraction Detector in local environment. Main notebook used for the experiments during the development of the project.
- `image2text.py`: Script to create a text dataset from the image dataset (using OCR).
- `LocalDatasetPreprocessing.ipynb`: Notebook to preprocess the local dataset (from raw to structured).
- `NLP_BinaryClassifier_demo.ipynb`: Demo notebook to load and test the Binary Classifier.
- `NLP_BinaryClassifier.ipynb`: Notebook to train and test the Binary Classifier model.
- `NLPutils.py`: Python module with the functions to preprocess the text data and apply NLP techniques for topic modeling.
- `OCRtest.py`: Script to test OCR techniques on screenshots.
- `README.md`: This file.
- `requirements.txt`: Python requirements file.
- `feature_logger.py`: Script to take screenshots and log the feature extraction (text and class).
- `image2text.py`: Script to create a text dataset from the image dataset (using OCR).
- `BinaryClassifier_demo.ipynb`: Notebook to test the Binary Classifier.
- `TopicModeling.ipynb`: Notebook showcasing the topic modeling process.

---

## Related Work

You can find the related work in the [related_work](/docs/related_work.md) file.

---

## Data

- Data is now stored in the shared folder in Google Drive.
- Screenshots should be full screen (size to be defined: e.g. 1280 x 720) and saved in JPG format using always the same format.
- The data is divided in 2 classes for now:
    - Focused
        - Ideas: Word, Excel, PowerPoint, Google Docs, Google Slides, Google Sheets, PDF, Google Drive, Coding, Microsoft Teams, Google Classroom, Google Translate, Calendar, Prezi, Edmodo, Wikipedia, Mentimeter, Miro, Moodle, Canvas [More ideas link](https://www.toptools4learning.com/)
            - Educational apps that look like games (Ideas: Scratch, Kahoot, Duolingo, Khan Academy, Socrative)
    - Distracted
        - Ideas: online shopping, news, Twitter, Facebook, Google Photos, Google Maps, Reddit, Pinterest, TikTok, Instagram, Streaming (Netflix, Prime Video, Disney+...)
            - Gaming (Ideas: Minecraft, Solitaire, Minesweeper, Agar.io, Slither.io, LoL, CSGO, Dino Game, online gamming platforms like Friv, Age of Empires, Epic Games, Steam, Akinator, Quick Draw, Catan Universe, Sudoku, Pacman, Wordle, Geoguessr, Freeciv-Web, War Brokers, Powerline.io, Skribbl, Diep.io. [More ideas Link](https://beebom.com/browser-games/))
- Potential classes that may cause problems:
    - Windows
    - Browser / Search Engine
    - Video / Tutorials
        - Ideas: Youtube, Twitch, Recorded classes in Drive
    - Chatting
        - Ideas: Whatsapp, Messenger, Web forum, Discord, Telegram, Groupme, Google Chat
        
    \
    Other ideas: Slack, Skype, Zoom, Google Meet, Google Hangouts, Facebook Messenger, Snapchat, LinkedIn, Tumblr, Quora, Stack Overflow, GitHub, GitLab, Bitbucket...

Some ideas for the data:
- [👁️ Interesting] We should consider using a data versioning tool like [DVC](https://dvc.org/) to keep track of the data and to make it easier to reproduce the results.
- We should discuss:
    - How many samples do we want to use for training and testing?
    - Can we do some data augmentation to increase the number of samples?

- [❌ Discarded] We could explore the possibility of scraping the data using automated image search from search engines like Google or Bing.

#### Data Transformations and Data Augmentation

When working with screenshots in a specific software with a controlled environment and fixed image format, common transformations such as crop and rotation may not be necessary and can even be detrimental to the model's performance. This is because in a controlled environment, the screenshots are likely to have consistent layouts and structures, and the objects of interest will be in the same position and orientation in each screenshot. As a result, cropping or rotating the images may cause the model to lose important information or even introduce noise, which can negatively impact its ability to make accurate predictions.

However, as with any application of Deep Learning for Computer Vision, it is still important to carefully evaluate the specific requirements of the problem at hand and determine whether additional transformations or techniques may be necessary for optimal performance. For example, if the screenshots may have variations in lighting or color, applying data augmentation techniques such as brightness adjustment or color jittering may help to improve the model's robustness and accuracy.

##### Resizing

- *Downsampling*: For now we are using `transforms.Resize(224)`. This keeps the proportion of the image (there is no point in changing the proportion) and lowers the size of the images (we have images of 1920x1080 and 3840x2160) making them all with same format (398x224). This is a common practice in CNNs. It may seem contraintuitive to feed the model with lower quality images, but models train faster on smaller images. An input image that is twice the size requires our network to learn from four times as many pixels, with more memory need and times that add up. [Reference](https://blog.roboflow.com/you-might-be-resizing-your-images-incorrectly/)

![img1](https://user-images.githubusercontent.com/71346949/225948123-11ed82f4-e9f4-4d9c-9051-5a3137c273ea.png) ![image](https://user-images.githubusercontent.com/71346949/225976126-06261c12-1024-4a1f-b492-fec5d61288ae.png)

---

## Experiment tracking

In the development of our machine learning project, we adopted the use of the [MLflow Python library](https://mlflow.org/). MLflow is a comprehensive platform designed to facilitate and streamline the process of machine learning development. It offers functionalities for experiment tracking, packaging code into reproducible runs, and enables sharing and deployment of models.
