import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Veriyi Yükle
df = pd.read_csv("data/intent_classification_dataset.csv")

# 2. Train / Test Ayrımı Yap
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['intent'], test_size=0.2, random_state=42
)

# 3. Pipeline Oluştur (TF-IDF + Lojistik Regresyon)
# Bu yöntem metni sayılara çevirir ve sınıflandırır
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# 4. Modeli Eğit
print("Model eğitiliyor...")
pipeline.fit(X_train, y_train)

# 5. Performansı Ölç (Ödev Kriteri: Precision, Recall, F1)
print("\n--- Model Performans Raporu ---")
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# Confusion Matrix (İsteğe bağlı kriter)
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# 6. Modeli Kaydet (Uygulamada kullanmak için)
joblib.dump(pipeline, 'intent_classificiation_model/intent_model.pkl')
print("\nModel 'intent_model.pkl' olarak kaydedildi.")