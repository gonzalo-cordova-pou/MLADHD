# MLADHD

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)

This repository contains the code for the project "Machine Learning for ADHD" at the Virginia Commonwealth University.

## TODOs

- [X] Add a README.md file to the repository
- [X] Agree on a Python version (suggestion: 3.10)
- [X] Migrate data to a cloud storage (suggestion: AWS S3 or Google Drive)
- [ ] Finish related work document
    - [ ] Summarize key takeaways
- [ ] Create ML Training System
    - [X] Create a notebook to train models
    - [X] Mount data in Google Colab from Google Drive
    - [X] Clone repo in Google Colab
    - [X] Set hyperparameters as global variables
    - [ ] Implement ML tracking tool (suggestion: MLFlow or WandB)
    - [ ] Energy Consumption Tracker [NVIDIA System Management Interface program](https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf)
    - [ ] Add new metrics: Precision and F1-Score
- [X] Modify `MLADHD` to support binary/multiclass classifier and freezed/unfreezed experiments
- [X] Create a notebook for wrong output analysis
- [X] Read: [Requirements Engineering for Machine Learning:
Perspectives from Data Scientists](https://arxiv.org/pdf/1908.04674.pdf)
- [X] Start Overleaf project

## Index

1. [New updates](#new-updates)
2. [Data](#data)
3. [Experiment guide](#experiment-guide)
4. [Installation](#installation)
5. [Related Work](#related-work)
---

## New updates

- **[15 Dec 2022]** I have created a notebook (`DEMO.ipynb`) to be executed in Google Colaboratory (you can open a GitHub file in Google Colab). This notebook mounts the Drive data (from the shared folder) and clones the repo. By doing this, we can train / test models in Google GPUs using the updated data from the Drive and the functions we code in the repo.
- **[07 Jan 2023]** I have made quite a few modifications to the `mlmodeling.py` file. These include a refactoring of code to handle experiments with a class. I have also parameterized all the hyperparameters so that I can initialize them with the experiment.
- **[07 Jan 2023]** I have created a new notebook (`DEMO_wandb.ipynb`) and made the necessary modifications to include experiment tracking with [Weights & Biases](https://wandb.ai/site). This is not the final version, but a preliminary to show how can it be used. WandB is a central dashboard to keep track of your hyperparameters, system metrics, and predictions so you can compare models live, and share your findings.
- **[23 Feb 2023]** We decided to work with a binary classifier for now. See [Experiment guide](#experiment-guide) for more info.
- **[09 Mar 2023]** Updates in the MLADHD class in `mlmodeling.py` file:
    - *load_model* method to load a model from pretrained weights
    - *predict* to make a prediction of a single image
    - *test_random_images* to make prediction of N random images and see results
- **[09 Mar 2023]** The (`DEMO.ipynb`) notebook has been updated with new sections:
    - A one image prediction
    - N random images prediction
    - Wrong predictions analysis

## Data

- Data is now stored in the shared folder in Google Drive.
- Screenshots should be full screen (size to be defined: e.g. 1280 x 720) and saved in JPG format using always the same format.
- The data is divided in 2 classes for now:
    - Work
        - Ideas: Word, Excel, PowerPoint, Google Docs, Google Slides, Google Sheets, PDF, Google Drive, Coding, Microsoft Teams, Google Classroom, Google Translate, Calendar, Prezi, Edmodo, Wikipedia, Mentimeter, Miro, Moodle, Canvas [More ideas link](https://www.toptools4learning.com/)
            - Educational apps that look like games (Ideas: Scratch, Kahoot, Duolingo, Khan Academy, Socrative)
    - Not Work
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

#### Resizing

- *Downsampling*: For now we are using `transforms.Resize(224)`. This keeps the proportion of the image (there is no point in changing the proportion) and lowers the size of the images (we have images of 1920x1080 and 3840x2160). This is a common practice in CNNs. It may seem contraintuitive to feed the model with lower quality images, but models train faster on smaller images. An input image that is twice the size requires our network to learn from four times as many pixels, with more memory need and times that add up. [Reference](https://blog.roboflow.com/you-might-be-resizing-your-images-incorrectly/)

## Possible OCR (optical character recognition) Python Package 

- [pytesseract](https://pypi.org/project/pytesseract/)
- [OpenCV](https://pypi.org/project/opencv-python/)

## Experiment guide

 1. The first set of experiments will be focused in training a binary classifier.
    - Performance metrics analysis
    - Wrong output analysis
 2. Based on the results above we will decide on:
    - Creating new classes (multiclass classifier instead of binary)
    - Introducing new ML techniques and features (eg. OCR + NLP)
 3. Redo the experiments with the new changes
    - Performance metrics analysis
    - Wrong output analysis

## Installation

It is recommended to use a virtual environment to avoid problems with libraries and dependencies. It is also important to agree on a Python version.

For Linux and MacOS:

```bash
pip install virtualenv # if not installed
python3 -m venv /path/to/new/virtual/environment
source venv/bin/activate
deactivate # to deactivate the virtual environment
```

For Windows:

```bash
pip install virtualenv # if not installed
virtualenv myenv # create a virtual environment in the current directory
myenv\Scripts\activate
deactivate # to deactivate the virtual environment
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Add required libraries to requirements.txt:

```bash
pip freeze > requirements.txt
```

## Related Work

You can find the related work in the [related_work](/docs/related_work.md) file.

---
