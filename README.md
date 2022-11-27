# MLADHD

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)

This repository contains the code for the project "Machine Learning for ADHD" at the Virginia Commonwealth University.

## TODOs

- [X] Add a README.md file to the repository
- [ ] Agree on a Python version (suggestion: 3.10)
- [ ] Migrate data to a cloud stroage (suggestion: AWS S3 or Google Drive)
- [ ] Document related work

## Index

1. [Data](#data)
2. [Installation](#installation)
3. [Related Work](#related-work)
---

## Data

Some ideas for the data:
- As mentioned in the TODOs, we should migrate the data to a cloud storage. This will make it easier to share the data with the team and to access it from different machines.
- We should also consider using a data versioning tool like [DVC](https://dvc.org/) to keep track of the data and to make it easier to reproduce the results.
- We should discuss:
    - How many classes do we want to classify?
    - How many samples do we want to use for training and testing?
    - Can we do some data augmentation to increase the number of samples?
    - We should agree on a file format for the images (e.g. PNG, JPG, etc.)

- We could explore the possibility of scraping the data using automated image search from search engines like Google or Bing.


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
