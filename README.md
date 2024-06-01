# Heart Disease Prediction

This repository hosts a Python project developed to predict heart disease using machine learning techniques. The project leverages various algorithms and data preprocessing methods to create a model that can assess the likelihood of heart disease based on input medical data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

Heart disease is one of the leading causes of death globally. Early prediction and diagnosis can significantly improve treatment outcomes. This project aims to predict heart disease using a machine learning model trained on medical data.

## Features

- Data preprocessing and cleaning
- Exploratory data analysis
- Machine learning model training and evaluation
- Predictions based on new data inputs
- Visualization of model performance

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dyingpotato890/Heart-Disease-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Heart-Disease-Prediction
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
OR

1. Please visit the following link to use the model through a simple user interface:
   [Make Predictions](https://heart-disease-prediction-dyingpotato890.streamlit.app/)

## Usage

1. Prepare your dataset and ensure it is in the correct format.
2. Run the preprocessing script to clean and prepare the data.
3. Train the machine learning model using the provided training script.
4. Use the trained model to make predictions on new data.

OR

1. [Click here to access the website for direct predictions.](https://heart-disease-prediction-dyingpotato890.streamlit.app/)

## Data

The dataset used in this project is the Heart Disease dataset from Kaggle. It includes various medical attributes relevant to heart disease diagnosis. This dataset consists of 1190 instances with 11 features. 

[Link To Dataset](https://www.kaggle.com/datasets/mexwell/heart-disease-dataset?)


## Model

The project employs several machine learning algorithms, including:

- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The Random Forest model gave the best performance and was selected based on accuracy (0.9411), precision, and recall.

## Results

The results of the project are evaluated using a test dataset. The performance of each model is visualized through a confussion matrix. The best-performing model is saved and can be used for future predictions. The model has been deploed using streamlit for easy public access.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please create a pull request or open an issue.
