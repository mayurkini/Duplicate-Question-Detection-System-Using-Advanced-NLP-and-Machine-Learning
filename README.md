# Duplicate-Question-Detection-System-Using-Advanced-NLP-and-Machine-Learning

## Overview
This project, developed in Jupyter Notebook format, focuses on detecting duplicate questions on Quora using various natural language processing (NLP) techniques and machine learning models. By leveraging custom features, advanced text preprocessing, and feature engineering, the project aims to improve the accuracy of identifying whether two given questions are duplicates.

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Data Preprocessing](#data-preprocessing)
- [Training Techniques](#training-techniques)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Objective
The goal is to build a machine learning system in a Jupyter Notebook that identifies duplicate question pairs on Quora, streamlining content moderation and improving search results on the platform.

## Dataset
The dataset used is the Quora Question Pairs dataset, which includes:
- `qid1`, `qid2`: Unique IDs for the questions.
- `question1`, `question2`: The actual questions in text form.
- `is_duplicate`: A binary indicator of whether the questions are duplicates (1 for duplicates, 0 for non-duplicates).

The dataset is available on [Kaggle](https://www.kaggle.com/c/quora-question-pairs/data).

## Project Structure
<pre>
├── data/ # Folder containing the dataset 
├── notebooks/ # Jupyter notebook for analysis and modeling
│        └── Quora_Duplicate_Question_Detection.ipynb 
├── models/ 
├── src/
├── requirements.txt # List of dependencies 
└── README.md # Project documentation
</pre>

## Model Architectures
The project uses the following models to predict whether two questions are duplicates:
- **Naive Bayes**
- **Logistic Regression**
- **Random Forest**
- **Decision Trees**

Models are evaluated after using both **Count Vectorization** and **TF-IDF Vectorization**.

## Data Preprocessing
### Cleaning Techniques:
- **BeautifulSoup** was used to clean the text by removing HTML tags, punctuation, and stopwords.
- Invalid rows (e.g., questions with fewer than 5-6 characters or blank questions) were removed.

### Custom Feature Engineering:
- **Basic Features**: Number of words, question lengths, common words, and word-sharing ratios between questions.
- **Advanced Features**:
  - **Token Features** (e.g., `cwc_min`, `csc_min`, `last_word_eq`).
  - **Length-Based Features** (e.g., `mean_len`, `abs_len_diff`).
  - **Fuzzy Features** using FuzzyWuzzy (e.g., `fuzz_ratio`, `token_set_ratio`).

### Feature Visualization:
KDE (Kernel Density Estimation) plots were generated for both duplicate and non-duplicate classes to visualize the effectiveness of the features.

## Training Techniques
- **Count Vectorizer**: Applied with a feature limit of 3000.
- **TF-IDF Vectorizer**: Also used with a feature limit of 3000.
- **Train-Test Split**: The dataset was split into training and test sets for model evaluation.

## Evaluation
The project evaluates model performance using:
- **Accuracy**
- **F1 Score**
- **Precision & Recall**
- **Confusion Matrix**

## Results
The initial models yielded accuracy between **70-75%**. After incorporating advanced features (token-based, length-based, and fuzzy features), accuracy improved to **80-81%**.

## Usage
### Running the Notebook:
1. Clone the repository:
   <pre>
   git clone https://github.com/your-username/quora-duplicate-detection.git</pre>

### Navigate to the project folder:
<pre>cd quora-duplicate-detection</pre>

### Install dependencies:
<pre>pip install -r requirements.txt</pre> 

### Open jupyter Notebook
<pre> jupyter notebook notebooks/Quora_Duplicate_Question_Detection.ipynb
</pre>

# Dependencies

The following Python libraries are required:

- `numpy`
- `pandas`
- `scikit-learn`
- `beautifulsoup4`
- `fuzzywuzzy`
- `matplotlib`, `seaborn` (for plotting)
- `notebook` (for running the Jupyter notebook)

Install dependencies using:

<pre>
pip install -r requirements.txt</pre>

## Troubleshooting

- **Kernel Crashes**: If the Jupyter kernel crashes due to memory issues, consider reducing the sample size or running the notebook on a machine with more RAM.
- **Slow Training Times**: Training time can be reduced by limiting the dataset size or using cloud-based GPUs for faster processing.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request for review.

