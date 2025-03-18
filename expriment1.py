# Import necessary libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Traditional ML classifiers
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Transformer-based methods from Hugging Face
from transformers import pipeline

# Create a progress bar
from tqdm import tqdm

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
# Load the cleaned data (make sure 'steam_reviews_cleaned.csv' is in your working directory)
df = pd.read_csv('steam_reviews_cleaned.csv')

# the output of each model 
results = {   'f1':{'RF': [], 'XGB': [], 'LR': []},
              'precision':{'RF': [], 'XGB': [], 'LR': []},
              'recall':{'RF': [], 'XGB': [], 'LR': []},
              'accuracy':{'RF': [], 'XGB': [], 'LR': []}}

# experiment the result between the test size and the f1, precision, recall, accuracy
for testsize in tqdm([0.1, 0.2, 0.3, 0.4, 0.5]):
    # Prepare the data for classification

    # The target is the 'voted_up' column (convert it to integer: 1 for positive, 0 for negative)
    X = df['cleaned_review']
    y = df['voted_up'].astype(int)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=42)

    print('Training set size:', len(X_train))
    print('Test set size:', len(X_test))
    # Vectorize the text using TF-IDF with n-grams (unigrams and bigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print('TF-IDF features shape:', X_train_tfidf.shape)
    # Train a Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_tfidf, y_train)

    # Predict and evaluate on the test set
    pred_rf = rf.predict(X_test_tfidf)
    #print('Random Forest Accuracy:', accuracy_score(y_test, pred_rf))
    #print(classification_report(y_test, pred_rf))
    results['f1']['RF'].append(classification_report(y_test, pred_rf, output_dict=True)['1']['f1-score'])
    results['precision']['RF'].append(classification_report(y_test, pred_rf, output_dict=True)['1']['precision'])
    results['recall']['RF'].append(classification_report(y_test, pred_rf, output_dict=True)['1']['recall'])
    results['accuracy']['RF'].append(accuracy_score(y_test, pred_rf))
    # Train an XGBoost classifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train_tfidf, y_train)

    # Predict and evaluate on the test set
    pred_xgb = xgb.predict(X_test_tfidf)
    #print('XGBoost Accuracy:', accuracy_score(y_test, pred_xgb))
    #print(classification_report(y_test, pred_xgb))
    results['f1']['XGB'].append(classification_report(y_test, pred_xgb, output_dict=True)['1']['f1-score'])
    results['precision']['XGB'].append(classification_report(y_test, pred_xgb, output_dict=True)['1']['precision'])
    results['recall']['XGB'].append(classification_report(y_test, pred_xgb, output_dict=True)['1']['recall'])
    results['accuracy']['XGB'].append(accuracy_score(y_test, pred_xgb))
    # Train a Logistic Regression classifier
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)

    # Predict and evaluate on the test set
    pred_lr = lr.predict(X_test_tfidf)
    # print('Logistic Regression (K-grams) Accuracy:', accuracy_score(y_test, pred_lr))
    # print(classification_report(y_test, pred_lr))
    results['f1']['LR'].append(classification_report(y_test, pred_lr, output_dict=True)['1']['f1-score'])
    results['precision']['LR'].append(classification_report(y_test, pred_lr, output_dict=True)['1']['precision'])
    results['recall']['LR'].append(classification_report(y_test, pred_lr, output_dict=True)['1']['recall'])
    results['accuracy']['LR'].append(accuracy_score(y_test, pred_lr))
    # print the result

# create a plt to show the result
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
metrics = list(results.keys())
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
for i, key in enumerate(metrics):
    row = i // 2
    col = i % 2
    for model in results[key].keys():
        ax[row, col].plot(test_sizes, results[key][model], label=model)
    ax[row, col].set_title(key)
    ax[row, col].set_xlabel('Test size')
    ax[row, col].set_ylabel(key)
    ax[row, col].legend()
plt.tight_layout()
plt.show()
