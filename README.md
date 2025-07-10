# ✈️ Airline Customer Sentiment Analysis using AI

## 📌 Problem Statement

Airlines receive thousands of customer reviews and social media mentions every day. Understanding the sentiment behind this textual feedback is critical to improving customer service, enhancing brand reputation, and making informed operational decisions.

## 🎯 Objective

This project aims to analyze customer sentiment from a dataset of airline tweets using natural language processing (NLP) techniques and build models that can classify tweets into Positive, Negative, or Neutral categories.

---

## 🧾 Dataset Description

- **File**: `US_Airways.csv`
- **Columns**:
  - `tweet_id`
  - `airline_sentiment`
  - `text`
  - `airline`
  - `retweet_count`
  - `tweet_created`
  - `negativereason` (optional)

---

## 🛠️ Technologies Used

- Python 3.x
- pandas, NumPy
- NLTK, scikit-learn
- WordCloud, seaborn, matplotlib
- Logistic Regression, Naive Bayes, Random Forest

---

## 🔁 Workflow

```python
# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 2. Load Dataset
df = pd.read_csv('US_Airways.csv')

# 3. Basic EDA
df['airline_sentiment'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Sentiment Distribution')
plt.show()

# 4. Text Preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# 5. Feature Extraction
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['airline_sentiment']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Training & Evaluation

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Naive Bayes:\n", classification_report(y_test, y_pred_nb))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred_rf))
```

---

## ✅ Results

- All models were able to distinguish between positive, neutral, and negative tweets.
- Logistic Regression performed well with balanced accuracy and precision.
- Random Forest showed robustness, especially in handling misclassified data.

---

## 📈 Potential Improvements

- Implement deep learning using LSTM for better context awareness.
- Fine-tune vectorization using n-grams and TF-IDF.
- Explore BERT-based sentiment classifiers for improved accuracy.

---

## 📁 Repository Structure

```
├── AI_Application_Case_Study_Airline_Customer_Sentiment_Analysis.ipynb
├── US_Airways.csv
├── README.md
```

---

## 👨‍💻 Author

**Suhaib Khalid**  
AI & NLP Enthusiast
