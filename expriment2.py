import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Traditional ML classifiers
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# For resampling (SMOTE)
from imblearn.over_sampling import SMOTE

# Progress bar for loops
from tqdm import tqdm

# Load the cleaned data (ensure 'steam_reviews_cleaned.csv' is in your working directory)
df = pd.read_csv('steam_reviews_cleaned.csv')

# Prepare the data: target is 'voted_up' (convert to int: 1 for positive, 0 for negative)
X = df['cleaned_review']
y = df['voted_up'].astype(int)

# Split the data into training and test sets (fixed test size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorize the text using TF-IDF with unigrams and bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Check the class distribution in the training set
pos_count = sum(y_train == 1)
neg_count = sum(y_train == 0)
print("Training set class distribution:")
print("Positive:", pos_count, "Negative:", neg_count)

# Compute scale_pos_weight for XGBoost when using class weighting
scale_pos_weight = neg_count / pos_count

# Define the balancing methods to compare
balancing_methods = ['None', 'class_weight', 'SMOTE']

# Initialize a dictionary to store the accuracy results
results = {}

for method in balancing_methods:
    results[method] = {}
    
    # Prepare the training data based on the balancing method
    if method == 'SMOTE':
        # SMOTE requires dense arrays, so convert the sparse TF-IDF matrix to dense
        X_train_dense = X_train_tfidf.toarray()
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_dense, y_train)
    else:
        X_train_bal, y_train_bal = X_train_tfidf, y_train
    
    # Iterate over the classifiers
    for clf_name in ['RF', 'XGB', 'LR']:
        if clf_name == 'RF':
            # For RandomForest, use class_weight if specified
            if method == 'class_weight':
                clf = RandomForestClassifier(random_state=42, class_weight='balanced')
            else:
                clf = RandomForestClassifier(random_state=42)
                
        elif clf_name == 'XGB':
            # For XGBoost, use scale_pos_weight for imbalance handling if using class weighting
            if method == 'class_weight':
                clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                    random_state=42, scale_pos_weight=scale_pos_weight)
            else:
                clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                
        elif clf_name == 'LR':
            # For Logistic Regression, pass class_weight when requested
            if method == 'class_weight':
                clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            else:
                clf = LogisticRegression(max_iter=1000, random_state=42)
        
        # Train the classifier
        # (For SMOTE, the training data is dense; for others, it remains sparse.)
        clf.fit(X_train_bal, y_train_bal)
        
        # Predict on the (untouched) test set
        pred = clf.predict(X_test_tfidf)
        acc = accuracy_score(y_test, pred)
        results[method][clf_name] = acc
        
        print(f"Method: {method}, Classifier: {clf_name}, Accuracy: {acc:.4f}")

# Plot the results for visual comparison
methods = list(results.keys())
classifiers = ['RF', 'XGB', 'LR']
for clf in classifiers:
    acc_scores = [results[method][clf] for method in methods]
    plt.plot(methods, acc_scores, marker='o', label=clf)

plt.xlabel("Balancing Method")
plt.ylabel("Accuracy")
plt.title("Impact of Balancing Techniques on Classifier Performance")
plt.legend()
plt.show()
