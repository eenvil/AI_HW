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
results = {'RF':[], 'XGB':[], 'LR':[]}

# experiment the result between the test size and the accuracy
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
    results['RF'].append(accuracy_score(y_test, pred_rf))
    # Train an XGBoost classifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train_tfidf, y_train)

    # Predict and evaluate on the test set
    pred_xgb = xgb.predict(X_test_tfidf)
    #print('XGBoost Accuracy:', accuracy_score(y_test, pred_xgb))
    #print(classification_report(y_test, pred_xgb))
    results['XGB'].append(accuracy_score(y_test, pred_xgb))
    # Train a Logistic Regression classifier
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)

    # Predict and evaluate on the test set
    pred_lr = lr.predict(X_test_tfidf)
    # print('Logistic Regression (K-grams) Accuracy:', accuracy_score(y_test, pred_lr))
    # print(classification_report(y_test, pred_lr))
    results['LR'].append(accuracy_score(y_test, pred_lr))

# create a plt to show the result
import matplotlib.pyplot as plt
plt.plot([0.1, 0.2, 0.3, 0.4, 0.5], results['RF'], label='RF')
plt.plot([0.1, 0.2, 0.3, 0.4, 0.5], results['XGB'], label='XGB')
plt.plot([0.1, 0.2, 0.3, 0.4, 0.5], results['LR'], label='LR')
plt.xlabel('Test Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


    