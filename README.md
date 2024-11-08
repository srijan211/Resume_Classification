# Resume_Classification
This project demonstrates a machine learning approach to classify resumes into various categories using Natural Language Processing (NLP) techniques and classification algorithms. We perform data preprocessing, feature extraction, and model evaluation for accurate resume classification.


## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Libraries Used](#libraries-used)
3. [Project Workflow](#project-workflow)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Modeling and Results](#modeling-and-results)
6. [Conclusion](#conclusion)

---

## Dataset Overview

The dataset used in this project is resume_dataset.csv, consisting of resumes categorized into different job roles. Each resume has a brief text description and a category (job role) to which it belongs.

Columns:
Category: The category label of each resume, representing job roles.
Resume: The resume text.
Structured Resume: A cleaned version of the resume text for model training.
Dataset Summary:
Total Records: 962
Total Categories: 25
The dataset requires substantial text preprocessing to ensure that only meaningful words contribute to the model’s training.


## Libraries Used

The following libraries are used in this project for data manipulation, visualization, text preprocessing, feature extraction, and machine learning modeling:

Pandas: For data loading and manipulation.
Numpy: For numerical operations.
Matplotlib & Seaborn: For data visualization to understand the category distribution.
Plotly: For creating interactive visualizations.
WordCloud: For visualizing the most frequent words in resumes.
Scikit-learn (sklearn): For text feature extraction (TF-IDF), model training (K-Nearest Neighbors classifier), and evaluation.
NLTK: For natural language processing tasks, including stop word removal and word tokenization.


## Project Workflow

This project is structured into the following steps:

Data Loading: Load the dataset and inspect its structure.
Data Cleaning: Apply basic NLP techniques to clean resume text, including:
Removing URLs, mentions, punctuations, hashtags, non-ASCII characters, and extra whitespace.
Exploratory Data Analysis (EDA): Visualize category distributions and identify frequent terms.
Feature Extraction: Convert resume text to a numerical representation using TF-IDF.
Model Training: Train a K-Nearest Neighbors (KNN) classifier using the processed resume data.
Evaluation: Evaluate the model on a test set and analyze performance using metrics like accuracy and classification report.


## Exploratory Data Analysis (EDA)

1. Category Distribution
We visualized the distribution of resumes across various job categories to identify any class imbalances.
2. Word Cloud of Resume Text
Using WordCloud, we generated a visualization showing the most common words across resumes, helping to identify important keywords.



## Modeling and Results

1. Text Preprocessing and Feature Extraction
To convert resume text into a format usable for machine learning, we used the TF-IDF Vectorizer from sklearn. This approach captures the importance of words based on their frequency across resumes.

2. Model Selection
We selected K-Nearest Neighbors (KNN) as the classifier for this task and used the One-vs-Rest strategy for multiclass classification.

3. Model Evaluation
We evaluated the model’s performance on both training and test sets using metrics like accuracy, precision, recall, and F1-score.

Results:
Training Accuracy: 99%
Test Accuracy: 99%

The classifier achieves high accuracy and precision across most categories, indicating that the model performs well in differentiating between job roles based on resume content.

## Conclusion

This project successfully demonstrates a text classification pipeline for resume categorization using NLP techniques and a KNN classifier. We achieve high accuracy through effective text preprocessing, feature extraction using TF-IDF, and robust model training. This model can be further optimized with additional NLP techniques or alternative classifiers for even better performance.
