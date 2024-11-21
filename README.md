# AWS Scholarship Projects

### 1. AWS Scholarship 1 - Project 2
In this project, users input parameters through the command line interface and argparse. These parameters include the data directory, checkpoint storage directory, model name, learning rate, the count of hidden units in the layers, the number of epochs, and the choice of utilizing GPU. Detailed information is displayed at each epoch, encompassing the current epoch number, accuracy, and running loss. Furthermore, users can evaluate checkpoints, obtaining the name of the identified item along with associated probabilities, providing a measure of certainty for each result.

dependencies for Machine learning application:
Establishing a virtual environment necessitates the incorporation of essential packages such as torch, JSON, numpy, PIL and matplotlib. This ensures a well-configured environment tailored to the requirements of the project.

If you want to show a preview of the application and how it works you can download the "preview.html" file

### 2. AWS Scholarship 2 - Project 1
## Overview
In this project, students will apply the knowledge and methods they learned in the Introduction to Machine Learning course to compete in a Kaggle competition using the AutoGluon library.

Students will create a Kaggle account if they do not already have one, download the Bike Sharing Demand dataset, and train a model using AutoGluon. They will then submit their initial results for a ranking.

After they complete the first workflow, they will iterate on the process by trying to improve their score. This will be accomplished by adding more features to the dataset and tuning some of the hyperparameters available with AutoGluon.

Finally they will submit all their work and write a report detailing which methods provided the best score improvement and why. A template of the report can be found [here](report-template.md).

To meet specifications, the project will require at least these files:
* Jupyter notebook with code run to completion
* HTML export of the jupyter notebbook
* Markdown or PDF file of the report

Images or additional files needed to make your notebook or report complete can be also added.

## Getting Started
* Clone this template repository `git clone git@github.com:udacity/nd009t-c1-intro-to-ml-project-starter.git` into AWS Sagemaker Studio (or local development).

<img src="img/sagemaker-studio-git1.png" alt="sagemaker-studio-git1.png" width="500"/>
<img src="img/sagemaker-studio-git2.png" alt="sagemaker-studio-git2.png" width="500"/>

* Proceed with the project within the [jupyter notebook](project-template.ipynb).
* Visit the [Kaggle Bike Sharing Demand Competition](https://www.kaggle.com/c/bike-sharing-demand) page. There you will see the overall details about the competition including overview, data, code, discussion, leaderboard, and rules. You will primarily be focused on the data and ranking sections.

### Dependencies

```
Python 3.7
MXNet 1.8
Pandas >= 1.2.4
AutoGluon 0.2.0 
```

### Installation
For this project, it is highly recommended to use Sagemaker Studio from the course provided AWS workspace. This will simplify much of the installation needed to get started.

For local development, you will need to setup a jupyter lab instance.
* Follow the [jupyter install](https://jupyter.org/install.html) link for best practices to install and start a jupyter lab instance.
* If you have a python virtual environment already installed you can just `pip` install it.
```
pip install jupyterlab
```
* There are also docker containers containing jupyter lab from [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html).

## Project Instructions

1. Create an account with Kaggle.
2. Download the Kaggle dataset using the kaggle python library.
3. Train a model using AutoGluon’s Tabular Prediction and submit predictions to Kaggle for ranking.
4. Use Pandas to do some exploratory analysis and create a new feature, saving new versions of the train and test dataset.
5. Rerun the model and submit the new predictions for ranking.
6. Tune at least 3 different hyperparameters from AutoGluon and resubmit predictions to rank higher on Kaggle.
7. Write up a report on how improvements (or not) were made by either creating additional features or tuning hyperparameters, and why you think one or the other is the best approach to invest more time in.

## License
[License](LICENSE.txt)

### 3. AWS Scholarship 2 - Project 2
In this project, I built a Handwritten Digit Classifier using PyTorch, achieving an impressive accuracy of 97.38%. The project was reviewed and approved by an AWS expert machine learning engineer.

<b>Key Highlights:</b>

PyTorch Mastery: Developed a deep understanding of PyTorch and its application in building neural networks.

Data Transformation: Applied various data transformations to enhance model performance.

Custom Neural Network: Designed, built, and trained a neural network from scratch to classify handwritten digits.

Overfitting Prevention: Implemented techniques such as early stopping to effectively prevent overfitting.

### 4. AWS Scholarship 2 - Project 3
In this project I developed a landmark image classifier. It was a challenging yet rewarding experience, simulating real-world applications in computer vision. The model, reviewed by an AWS ML engineer, showed significant learning and performance improvements.

### 5. AWS Scholarship 2 - Project 4
In project focuses on creating and deploying a comprehensive workflow using Amazon services to build and monitor the performance of a bike and motorcycle classifier.
