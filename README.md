# Data Science & Python Internship
## About the Internship
This repository contains the work I completed during my **Data Science & Python Internship with YUVA Intern**.
The internship was focused on understanding and applying the basic to intermediate concepts used in Data Science and Machine Learning. The tasks were completed progressively, starting with data collection and preprocessing and then moving towards exploratory data analysis, machine learning, clustering, deep learning, and finally an end-to-end capstone project.
I used Python throughout the internship and worked with different libraries and tools such as Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn and PyTorch.

The main purpose of maintaining this repository is to keep all my internship tasks, Python programs, analysis reports and final project work together in one place.
-----
## Internship Organization

Organization: YUVA Intern
Internship Domain: Data Science & Python
Duration: 6 Weeks
Role: Data Science-Python Intern
-----
## Internship Objectives
During the internship, I worked on the following areas:
* Python programming for data analysis
* Data acquisition and dataset understanding
* Data cleaning and preprocessing
* Handling missing values and duplicate records
* Exploratory Data Analysis (EDA)
* Data visualization
* Statistical and correlation analysis
* Feature engineering and feature selection
* Supervised Machine Learning
* Unsupervised Machine Learning
* Clustering and customer/data segmentation
* Model evaluation and validation
* Deep Learning using PyTorch
* End-to-end Data Science workflow
* Drawing useful insights from data
* Preparing technical reports and documenting the work
-----
# Weekly Tasks
## Week 1 – Data Acquisition, Cleaning and Preprocessing
### Objective
The first task was focused on understanding a dataset and preparing it for further analysis and machine learning.
I worked with the **UCI Adult / Census Income dataset** and performed the basic steps required before using a dataset for analysis or modeling.
### Work Completed
* Acquired and loaded the dataset using Python
* Inspected the dataset structure
* Checked rows, columns and data types
* Identified missing values
* Identified duplicate records
* Standardized text values
* Handled missing-value placeholders
* Performed basic data-quality checks
* Checked numerical ranges
* Reviewed possible outliers
* Encoded categorical variables
* Prepared the target variable
* Considered feature scaling and data leakage
* Validated the processed data
### Main Technologies
* Python
* Pandas
* NumPy
* Scikit-learn
### Files
The Week 1 folder contains the Python implementation and the detailed report explaining the complete preprocessing process.
-----
# Week 2 – Exploratory Data Analysis and Visualization
### Objective
The second task focused on understanding the dataset through exploratory analysis and visualizations.
The same dataset was continued from Week 1 so that the work followed a proper data science workflow instead of starting with a completely unrelated dataset.
### Work Completed
* Performed descriptive statistical analysis
* Studied numerical variables
* Studied categorical variables
* Analyzed the target variable
* Created distribution plots
* Compared different categories
* Studied education and income relationships
* Analyzed workclass and income
* Studied working hours
* Examined age-related patterns
* Created correlation analysis
* Performed multivariate analysis
* Created aggregated views for comparison
* Identified important patterns and unusual observations
* Documented the findings from the analysis
### Main Technologies
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

### Files
The Week 2 folder contains the EDA Python script and the complete analysis report.
-----
# Week 3 – Unsupervised Learning and Clustering
### Objective
The third task introduced unsupervised learning and clustering techniques.
For this task, I worked with the **Mall Customers dataset** and used customer-related attributes to identify groups with similar characteristics.
### Work Completed
* Loaded and inspected the customer dataset
* Selected relevant numerical features
* Removed the customer ID from modeling
* Prepared the data for clustering
* Standardized the selected features
* Applied the K-Means clustering algorithm
* Tested different numbers of clusters
* Used the Elbow Method
* Used Silhouette Score for cluster evaluation
* Selected an appropriate number of clusters
* Visualized the resulting clusters
* Studied cluster characteristics
* Compared customer segments
* Interpreted the clusters from a business perspective

### Main Technologies
* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* K-Means Clustering
### Files
The Week 3 folder contains the clustering implementation and the detailed report.
-----
# Week 4 – Supervised Learning Model Implementation
### Objective
The fourth task focused on supervised machine learning.
The objective was to understand how a machine learning model can be trained using labeled data and then evaluated on unseen data.
### Work Completed
* Selected a suitable public dataset
* Performed data inspection
* Cleaned and prepared the data
* Performed feature preprocessing
* Applied feature engineering where required
* Split the dataset into training and testing data
* Selected an appropriate supervised learning algorithm
* Trained the model using Scikit-learn
* Performed model validation
* Used cross-validation
* Evaluated model performance
* Studied different evaluation metrics
* Identified strengths and limitations of the model
* Considered possible improvements

### Main Technologies
* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

### Files
The Week 4 folder contains the supervised learning implementation and its detailed report.
---
# Week 5 – Deep Learning Application
### Objective
The fifth task introduced the use of Deep Learning for a practical Data Science problem.
For this task, I implemented a neural network using **PyTorch** for handwritten digit classification.
### Dataset
The project uses the **Scikit-learn Digits dataset**, which contains images of handwritten digits from 0 to 9.

### Work Completed
* Loaded the handwritten digit dataset
* Examined the image data
* Prepared and normalized the input features
* Divided the data into training, validation and testing sets
* Built a neural network using PyTorch
* Used fully connected layers
* Applied ReLU activation
* Used Dropout for regularization
* Used Cross-Entropy Loss
* Used the Adam optimizer
* Implemented model training
* Monitored training and validation performance
* Used early stopping
* Evaluated the final model
* Generated a confusion matrix
* Studied classification performance for individual classes
* Examined incorrectly classified samples

### Model
The neural network used the following general structure:

Input Layer
    ↓
128 Neurons
    ↓
ReLU
    ↓
Dropout
    ↓
64 Neurons
    ↓
ReLU
    ↓
Dropout
    ↓
10 Output Classes

### Final Result
The final model achieved approximately:
**97.41% test accuracy**
along with strong precision, recall and F1-score performance.

### Main Technologies
* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

### Files
The Week 5 folder contains the PyTorch implementation, metrics file and detailed report.
-----
# Week 6 – Integrative Capstone Project
## End-to-End Data Science Project
The sixth and final task was designed as an integration of the concepts covered throughout the internship.
Instead of treating this as another individual machine learning exercise, I used it to put together a complete Data Science workflow starting from the dataset and ending with model evaluation and insights.

### Project
End-to-End Data Science Analysis using the Breast Cancer Wisconsin Diagnostic Dataset
### Problem Statement
The objective of this project was to analyze a medical diagnostic dataset, understand the relationships between the available numerical features, build a supervised classification model and explore whether the observations could also be grouped using unsupervised learning.
This was done as an educational machine learning project to demonstrate the complete workflow. It is not intended to be used as a real-world medical diagnostic system.

### Dataset
The project uses the **Breast Cancer Wisconsin Diagnostic dataset** available through the Scikit-learn dataset collection.
The dataset contains numerical measurements calculated from digitized images of breast mass samples.
### Dataset Details
* 569 observations
* 30 numerical input features
* Binary classification problem
* Malignant and benign diagnosis categories

### Work Completed
The capstone combines the major areas covered during the internship:

Data Acquisition
       ↓
Data Inspection
       ↓
Data Cleaning & Quality Checks
       ↓
Exploratory Data Analysis
       ↓
Feature Analysis
       ↓
Feature Selection
       ↓
Supervised Learning
       ↓
Cross-Validation
       ↓
Model Evaluation
       ↓
Unsupervised Clustering
       ↓
Cluster Analysis
       ↓
Insights & Recommendations

### Data Preparation
The dataset was inspected for:
* Missing values
* Duplicate records
* Incorrect data types
* Feature distributions
* Target distribution
* Feature relationships
The dataset did not contain missing values or duplicate records in the final analysis.

### Exploratory Data Analysis
The EDA included:
* Diagnosis distribution
* Feature distributions
* Comparison of important features across diagnosis categories
* Correlation analysis
* Identification of strongly related variables
* Visualization of important patterns

### Supervised Learning
For the classification part, I used:

Logistic Regression:
with:
* StandardScaler
* SelectKBest feature selection
* Stratified train/test split
* 5-fold Stratified Cross-Validation
The preprocessing and feature-selection steps were kept inside the machine learning pipeline so that they could be applied correctly during cross-validation.

### Model Evaluation
The final model was evaluated using:
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Cross-validation results

The held-out test results were approximately:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 97.37% |
| Precision | 98.59% |
| Recall    | 97.22% |
| F1-Score  | 97.90% |
| ROC-AUC   | 99.44% |

The 5-fold cross-validation accuracy was approximately **97.58%**.

### Unsupervised Learning
I also applied **K-Means clustering** to selected numerical features.
Different values of K were tested and compared using:
* Elbow Method
* Silhouette Score
The analysis selected **2 clusters** based on the evaluation results.
The clustering results were then studied alongside the diagnosis labels to understand whether the discovered groups showed meaningful differences.

### Important Note
The diagnosis information was not used as an input feature for the K-Means model.
It was only used afterward to help interpret the resulting clusters.
-----

# Key Learning Outcomes
By completing the six tasks, I gained practical experience with the overall Data Science workflow.
Some of the main things I learned were:
* How to work with public datasets
* How to identify and handle data-quality problems
* How important preprocessing is before modeling
* How to explore data instead of directly training a model
* How visualizations can help identify patterns
* How supervised and unsupervised learning are different
* How to select and evaluate machine learning models
* Why cross-validation is useful
* How to build a basic neural network using PyTorch
* How to compare model performance using different metrics
* How clustering can be used to discover hidden groups
* How to document machine learning experiments
* How different stages of Data Science connect together
-----
# Technologies Used
### Programming
* Python
### Data Analysis
* Pandas
* NumPy
### Data Visualization
* Matplotlib
* Seaborn
### Machine Learning
* Scikit-learn
* Logistic Regression
* K-Means Clustering
* Feature Selection
* Cross-Validation
* StandardScaler
### Deep Learning
* PyTorch
### Development Tools
* Jupyter Notebook / Python
* VS Code
* Git
* GitHub
-----
# Repository Structure

data-science-python-internship/
│
├── README.md
├── requirements.txt
│
├── Week-1-Data-Acquisition-Cleaning/
│   ├── README.md
│   ├── week1_data_cleaning.py
│   └── Week_1_Report.docx
│
├── Week-2-EDA-Visualization/
│   ├── README.md
│   ├── week2_eda.py
│   └── Week_2_Report.docx
│
├── Week-3-Unsupervised-Learning/
│   ├── README.md
│   ├── week3_clustering.py
│   └── Week_3_Report.docx
│
├── Week-4-Supervised-Learning/
│   ├── README.md
│   ├── week4_supervised_learning.py
│   └── Week_4_Report.docx
│
├── Week-5-Deep-Learning/
│   ├── README.md
│   ├── week5_deep_learning_pytorch.py
│   ├── week5_deep_learning_metrics.csv
│   └── Week_5_Report.docx
│
└── Week-6-Integrative-Capstone/
    ├── README.md
    ├── week6_integrative_data_science_capstone.py
    ├── week6_capstone_metrics.csv
    ├── Week_6_Report.docx
    └── figures/
-----

## Acknowledgement
I would like to thank **YUVA Intern** for providing this internship opportunity and for giving me the chance to work on practical Data Science and Python tasks.
The internship provided a useful opportunity to apply concepts that I had studied theoretically and gain more hands-on experience with Python, data analysis and machine learning.
-----

## Disclaimer
The datasets and machine learning models used in this repository are intended for educational and learning purposes.
The Week 6 medical dataset project is an academic machine learning exercise and should not be considered a clinical diagnostic tool or used for real-world medical decisions.
-----

## Author
Vedang Raut
BE-IT Engineering Student
Data Science & Python Intern
