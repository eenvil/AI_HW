# Steam Reviews Data Collection and Analysis

## Dataset Web Link
[GitHub Repository](https://github.com/eenvil/AI_HW)

## Overview
This repository contains scripts for gathering, cleaning, and analyzing Steam reviews. The data collection process is divided into multiple steps, including data gathering, preprocessing, and experimentation with different machine learning models.

## Repository Contents

### Data Collection & Cleaning
- **d1.py**: Collects raw review data from the Steam platform.
- **d2.py**: Cleans the raw review data to produce a dataset suitable for analysis.
- **d3.ipynb**: Implements five methods to analyze the dataset, showcasing its utility and base score.
- **steam_reviews.csv**: The raw collected reviews dataset.
- **steam_reviews_cleaned.csv**: The cleaned dataset after preprocessing.

### Experiments
- **experiment1.py**: Examines the effect of training data size on model performance. It includes:
  - Splitting the dataset into various training and testing proportions.
  - Vectorizing text using TF-IDF (unigrams and bigrams).
  - Training multiple classifiers (Random Forest, XGBoost, and Logistic Regression).
  - Evaluating accuracy and generating visual plots to show performance trends.
  
- **experiment2.py**: Investigates the impact of data composition and balance on model performance. It includes:
  - Comparing different balancing strategies: original imbalanced dataset, class weighting, and SMOTE resampling.
  - Preprocessing data to convert it into a dense format when necessary.
  - Training the same classifiers under each balancing method.
  - Producing evaluation metrics and comparative plots.

## Usage
1. Run `d1.py` to gather raw review data.
2. Use `d2.py` to clean and preprocess the dataset.
3. Utilize `d3.ipynb` to explore various methods for initial analysis.
4. Execute `experiment1.py` to analyze how training size affects model performance.
5. Run `experiment2.py` to assess the effect of dataset balancing strategies.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install requests csv pandas numpy scikit-learn xgboost imbalanced-learn matplotlib langdetect nltk transformers
```

Additionally, download the necessary NLTK resources by running:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## License
This project is open-source and available under the MIT License.

## Contact
For any issues or inquiries, please open an issue on the repository.

