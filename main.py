import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# --- STEP 1: DATASET (Internal Data) ---
data = {
    'review': [
        "The acting was brilliant and the plot was very engaging.",
        "A true masterpiece, I highly recommend this film.",
        "The cinematography was breathtaking and beautiful.",
        "I loved every single second of this movie.",
        "The movie was not bad, actually it was quite good.",
        "The story was boring and the pacing was way too slow.",
        "Worst movie I have ever seen in my entire life.",
        "I hated the ending, it made no sense at all.",
        "A complete waste of time and money, do not watch.",
        "Not worth the ticket, the performance was terrible."
    ],
    'sentiment': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] # 1: Positive, 0: Negative
}

df = pd.DataFrame(data)

# --- STEP 2: FEATURE EXTRACTION (TF-IDF + BIGRAMS) ---
# Dùng ngram_range=(1, 2) để máy hiểu cụm từ như "not bad"
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# --- STEP 3: MODEL TRAINING (LOGISTIC REGRESSION) ---
model = LogisticRegression()
model.fit(X, y)

# --- STEP 4: EVALUATION & VISUALIZATION ---
y_pred = model.predict(X)
print(f"Model Accuracy: {accuracy_score(y, y_pred) * 100}%")

# Vẽ Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Sentiment Analysis')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

# Lưu ảnh để dán vào báo cáo
plt.savefig('confusion_matrix.png')
print("Confusion Matrix saved as 'confusion_matrix.png'")
plt.show()
