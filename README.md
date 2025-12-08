# IMDB Sentiment Classification Project
## PROG8245 – Fall 2025

## Group Members
| Name | Student ID |
|------|------------|
| Rohit Iyer | 8993045 |
| Jatindar Pal Singh | 9083762 |
| Cemil Caglar Yapici | 9081058 |


## Project Overview
his project performs binary text classification on the IMDB Movie Review Dataset from Stanford.
The goal is to classify movie reviews as positive or negative, using:
- TF-IDF vectorization
- Naive Bayes baseline classifier
- Dimensionality reduction (SVD & PCA)
- Logistic Regression models on reduced features
- Confusion matrix comparison across all models
- The project demonstrates how preprocessing, feature extraction, and dimensionality reduction impact classification performance.

## Dataset: IMDB Movie Review Dataset (Stanford)

- Total Reviews: 50,000
- Sentiment Labels: Positive (1), Negative (0)
- Source: Stanford AI Lab
- Source: https://ai.stanford.edu/~amaas/data/sentiment/

For this Project, we use only 2000 reviews.
- 2,000 reviews used (1,000 positive, 1,000 negative)
- Loaded using tensorflow_datasets

They are loaded using:

import tensorflow_datasets as tfds

(ds_train, ds_test), info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)


## How to Run

### 1. Install Dependencies
pip install nltk tensorflow tensorflow-datasets scikit-learn pandas numpy matplotlib seaborn

### 2. Download NLTK Data
import nltk
nltk.download('punkt')
nltk.download('stopwords')

### 3. Run the Notebook
jupyter notebook
Open Final_Project.ipynb

## Project Steps
1. Data Loading & Preprocessing  
- Load 2,000 IMDB reviews (1,000 pos + 1,000 neg)
- Store them in a DataFrame (df)
- Split into training and test sets
- Explore the distribution of both labels
- Apply text preprocessing:
    - Lowercasing
    - Tokenization
    - Optional: stopword removal

This step ensures that raw text is ready for feature extraction.

2. TF-IDF Feature Extraction  
- Convert text into TF-IDF vectors using TfidfVectorizer
- Understand how TF-IDF weights important vs common terms
- Inspect:
   - Vocabulary size
   - Example feature names
   - TF-IDF weights for a sample document
- Visualize:
   - TF-IDF matrix shape
   - Sparsity percentage
   - A small dense sample matrix

This step transforms text into machine-readable numeric form.

3. Baseline Naive Bayes Model  
- Train Multinomial Naive Bayes on TF-IDF features
- Predict on test data
- Generate a confusion matrix
- Visualize it with a heatmap
- Extract:
   - TP, TN, FP, FN
- Compute:
   - Accuracy
   - Precision
   - Recall
   - F1-score
- Interpretation Requirements:
   - How many samples were correctly classified?
   - Which errors are more common (FP or FN)?
   - Which class is harder for the model to predict?

This establishes the baseline model for comparison.

Sample Code:

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
cm_df = pd.DataFrame(cm,
                     index=["True Positive", "True Negative"],
                     columns=["Predicted Positive", "Predicted Negative"])

sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


4. SVD Dimensionality Reduction  
 -Apply Truncated SVD (Latent Semantic Analysis) to TF-IDF
- Reduce dimensionality from ~5000 features to 50–100 components
- Plot the explained variance ratio

Discuss in Markdown:
- How SVD uncovers semantic structure in text
- How lower dimensions affect data representation

This step makes features denser and potentially more informative.

5. Logistic Regression with SVD  
- Train Logistic Regression using SVD-reduced features
- Predict on test data
- Generate and visualize a confusion matrix
- Compare performance to Naive Bayes baseline

Discuss:
- Did reduction help or hurt accuracy?
- Was training faster?
- Why might SVD improve or degrade classification performance?

6. PCA Dimensionality Reduction 
- Standardize TF-IDF features (PCA requires scaling)
- Apply PCA to the same number of components as SVD
- Compare PCA’s variance curve to SVD

Markdown Discussion:
- Why PCA behaves differently from SVD in text data
- Which technique captures structure better

7. Logistic Regression with PCA  
- Train Logistic Regression on PCA-reduced data
- Predict on test set
- Generate a confusion matrix
- Compare accuracy against:
   - Naive Bayes baseline
   - Logistic Regression + SVD

Analysis:
- Which dimensionality reduction technique worked best and why?
- Does PCA distort sparse text data more than SVD?

8. Visual Comparison of Confusion Matrices  
You will create a side-by-side visualization of:
- Naive Bayes confusion matrix
- Logistic Regression + SVD confusion matrix
- Logistic Regression + PCA confusion matrix

This gives a clear visual summary of which model performs better.

## License
IMDB Dataset © Stanford AI Lab  
Code released under MIT License
