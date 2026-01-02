🎬 MovieBot
MovieBot, film severler için geliştirilmiş, Kaggle'dan https://www.kaggle.com/datasets/hrishabhtiwari/imdb-top-1000-movies-dataset bağlantısı üzerinden indirdiğim IMDB Top 1000 veri setini ve Google Gemini yapay zekasını kullanan hibrit bir sohbet asistanıdır.

Bu proje, maliyet ve performansı optimize etmek için Intent Sınıflandırma (Niyet Analizi) ile RAG (Retrieval-Augmented Generation) mimarisini bir arada kullanır. Basit sohbetler yerel bir modelle, karmaşık film sorguları ise Gemini AI ile yanıtlanır.

## Özellikler

- 🧠 Hibrit Yapı:
      - **Niyet Analizi:** Kullanıcının amacını (Selamlaşma, Film Sorusu vb.) yerel bir modelle (scikit-learn) tespit eder.
      - **RAG Motoru:** Film soruları için vektör veritabanından bağlam (context) çeker.
- **📂 Modüler Yapı:** Veriler, model eğitimi ve uygulama mantığı ayrı klasörlerde organize edilmiştir.
- **🔍 Vektör Arama:** ChromaDB kullanarak filmler arasında anlamsal arama yapar.
- **🤖 Google Gemini:** Doğal dil işleme ve cevap üretimi için gemini-2.5-flash-lite modelini kullanır.

## Ön Şartlar

- Python 3.8+
- Google API Key (LLM ve Embeddings için)
- IMDB Top 1000 Movies Dataset (CSV dosyası)

## 📂 Proje Yapısı
Proje dosyaları aşağıdaki dizin yapısına göre organize edilmiştir:

GEMINI-STREAMLIT-MOVIEBOT/
├── app/
│   ├── app.py                            # Ana Streamlit uygulama dosyası
│   └── Moviebot.ipynb                    # Geliştirme ve test not defteri (Notebook)
├── data/
│   ├── IMDb_Top_1000_Movies_Dataset.csv  # Film veri seti (Kaynak)
│   └── intent_classification_data.csv    # Niyet sınıflandırma modeli eğitim verisi
├── intent_classification_model/
│   ├── intent_model.pkl                  # Eğitilmiş niyet sınıflandırma modeli
│   └── train_classifier.py               # Modeli yeniden eğitmek için kullanılan script
├── .env                                  # API anahtarları (Gizli dosya)
├── .gitignore                            # Git tarafından göz ardı edilecek dosyalar
├── README.md                             # Proje dokümantasyonu
└── requirements.txt                      # Gerekli kütüphaneler

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

1. Repoyu klonla:
```bash
git clone https://github.com/nihalm-collab/gemini-basic-example.git
cd moviebot
```
2. Miniconda veya Anaconda kurulu ise, proje için yeni bir sanal ortam oluşturun:

```bash
# Python 3.10 tabanlı yeni bir ortam oluştur
conda create --name rag_env python=3.10

# Ortamı aktive et
conda activate tag_env
```

3. Gerekli paketleri yükle:

```bash
pip install -r requirements.txt
```

4. `.env` dosyası oluştur ve API key'ini ekle:
```
GOOGLE_API_KEY=your_google_api_key_here
```

5. `IMDB_Top_1000_Movies_Dataset.csv` veri setinin proje dizi içerisinde olduğundan emin ol.

## Kullanım

1. Streamlit'le çalıştır
```bash
streamlit run app/app.py
```

2. Tarayıcı açılır (`http://localhost:8501`)

3. İstersen selamlaş, vedalaş veya filmler ile ilgili sorularını gir:
   - "Recommend me some action movies?"
   - "What is the plot of Titanic?"
   - "Show me comedy movies starring Meryl Streep"
   - "What is the cast of Schindeler's List?"

## 🧠 Model Eğitimi (Opsiyonel)
Eğer niyet sınıflandırma modelini güncellemek veya yeni verilerle tekrar eğitmek isterseniz:

1. data/intent_classification_data.csv dosyasını güncelleyin.

2. Ana dizinde şu komutu çalıştırın:

```bash
python intent_classification_model/train_classifier.py
```

## 🛠️ Kullanılan Teknolojiler

- Python 3.10

- Streamlit (Arayüz)

- LangChain (RAG Orkestrasyonu)

- Google Gemini API (LLM & Embeddings)

- ChromaDB (Vektör Veritabanı)

- Scikit-Learn (Niyet Sınıflandırma)

- Miniconda (Ortam Yönetimi)

