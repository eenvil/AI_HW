import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# For resampling
from imblearn.over_sampling import SMOTE

from tqdm import tqdm

# Load the cleaned dataset
df = pd.read_csv('steam_reviews_cleaned.csv')
X = df['cleaned_review']
y = df['voted_up'].astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorize text using TF-IDF (unigrams and bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Compute scale_pos_weight for XGBoost (useful for 'class_weight')
neg_count = sum(y_train == 0)
pos_count = sum(y_train == 1)
scale_pos_weight = neg_count / pos_count

# Balancing methods to experiment with
balancing_methods = ['None', 'class_weight', 'SMOTE']

# Dictionary to store F1, Precision, Recall, Accuracy for each method/model
results = {
    'f1':        {'RF': [], 'XGB': [], 'LR': []},
    'precision': {'RF': [], 'XGB': [], 'LR': []},
    'recall':    {'RF': [], 'XGB': [], 'LR': []},
    'accuracy':  {'RF': [], 'XGB': [], 'LR': []}
}

for method in tqdm(balancing_methods):
    # Depending on the balancing method, prepare the training data
    if method == 'SMOTE':
        # SMOTE requires a dense representation
        X_train_dense = X_train_tfidf.toarray()
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_dense, y_train)
    else:
        X_train_bal, y_train_bal = X_train_tfidf, y_train

    # Train and evaluate models
    for clf_name in ['RF', 'XGB', 'LR']:
        # Instantiate classifier
        if clf_name == 'RF':
            if method == 'class_weight':
                clf = RandomForestClassifier(random_state=42, class_weight='balanced')
            else:
                clf = RandomForestClassifier(random_state=42)

        elif clf_name == 'XGB':
            if method == 'class_weight':
                clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                    random_state=42, scale_pos_weight=scale_pos_weight)
            else:
                clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        elif clf_name == 'LR':
            if method == 'class_weight':
                clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            else:
                clf = LogisticRegression(max_iter=1000, random_state=42)

        # Fit the model
        clf.fit(X_train_bal, y_train_bal)

        # Predict on the (untouched) test set
        pred = clf.predict(X_test_tfidf)

        # Extract classification metrics for the positive class (label "1")
        report = classification_report(y_test, pred, output_dict=True)
        f1 = report['1']['f1-score']
        precision = report['1']['precision']
        recall = report['1']['recall']
        accuracy = accuracy_score(y_test, pred)

        # Store results
        results['f1'][clf_name].append(f1)
        results['precision'][clf_name].append(precision)
        results['recall'][clf_name].append(recall)
        results['accuracy'][clf_name].append(accuracy)

# Plot the results for each metric and classifier
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
metrics = list(results.keys())
bal_methods = balancing_methods

for i, metric in enumerate(metrics):
    row = i // 2
    col = i % 2
    for model in results[metric].keys():
        ax[row, col].plot(bal_methods, results[metric][model], marker='o', label=model)
    ax[row, col].set_title(metric.capitalize())
    ax[row, col].set_xlabel('Balancing Method')
    ax[row, col].set_ylabel(metric)
    ax[row, col].legend()
    
plt.tight_layout()
plt.show()
