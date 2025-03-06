# Random Forest Classification

## **Overview**

This repository contains a Python script that implements a **Random Forest Classification model** using the **scikit-learn** library. The script loads a dataset, preprocesses the data, trains a **Random Forest model**, evaluates its performance, and saves the results.

This implementation is designed to efficiently handle classification tasks by using an ensemble learning approach. The script standardizes features, splits the dataset into training and testing sets, and provides key performance metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.

## **What is Random Forest?**

Random Forest is a powerful **ensemble learning algorithm** that builds multiple decision trees and combines their outputs to improve accuracy and robustness. It is widely used for **classification and regression tasks** due to its ability to handle large datasets, reduce overfitting, and capture complex patterns.

### **Key Concepts of Random Forest:**
- **Bagging (Bootstrap Aggregating):** Multiple subsets of the data are randomly selected, and separate decision trees are trained on these subsets.
- **Feature Randomness:** Each decision tree in the forest considers only a random subset of features, reducing correlation between trees.
- **Majority Voting:** In classification tasks, the final prediction is determined by the majority vote of all decision trees.

## **Setup and Usage**

### **1. Clone the Repository**
- **To get started, clone this repository to your local machine:**
  ```bash
  git clone https://github.com/yourusername/Random-Forest-Classifier.git
  cd Random-Forest-Classifier

### **2. Install Dependencies**
Ensure you have Python 3.8+ installed. Install the required dependencies.

### **3. Prepare Your Dataset**
Place your dataset in the data/ directory.
Ensure the dataset is in CSV format and contains a column named "Class" (target variable).
Update the DATASET_PATH variable in RF_Training_Testing.py to point to your dataset.

### **4. Run the Script**
- **Execute the script to train and evaluate the model:**
  ```bash
  python RF_Training_Testing.py

### **5. View the Results**
- **The output metrics, including accuracy, precision, recall, F1-score, and AUC-ROC, will be saved in: output_results.csv**

### **References**
Breiman, L. (2001). Random forests. Machine learning, 45, 5-32.
