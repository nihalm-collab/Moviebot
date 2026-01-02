# 🎬 MovieBot RAG

**MovieBot**, film severler için geliştirilmiş, kaggle'dan https://www.kaggle.com/datasets/hrishabhtiwari/imdb-top-1000-movies-dataset bağlantısı üzerinden indirilen **IMDB_Top_1000_Movies.csv** veri setini ve **Google Gemini** yapay zekasını kullanan hibrit bir sohbet asistanıdır.

Bu proje, maliyet ve performansı optimize etmek için **Intent Sınıflandırma** (Niyet Analizi) ile **RAG (Retrieval-Augmented Generation)** mimarisini bir arada kullanır. Basit sohbetler yerel bir modelle, karmaşık film sorguları ise Gemini AI ile yanıtlanır.

## 🚀 Özellikler

* **🧠 Hibrit Yapı:**
    * **Niyet Analizi:** Kullanıcının amacını (Selamlaşma, Film Sorusu vb.) yerel bir modelle tespit eder.
    * **RAG Motoru:** Film soruları için vektör veritabanından bağlam çeker.
* **📂 Modüler Yapı:** Veriler, model eğitimi ve uygulama mantığı ayrı klasörlerde organize edilmiştir.
* **🔍 Vektör Arama:** `ChromaDB` kullanarak filmler arasında anlamsal arama yapar.
* **🤖 Google Gemini:** Doğal dil işleme ve cevap üretimi için `gemini-2.5-flash-lite` modelini kullanır.

## 📂 Proje Yapısı

Proje dosyaları aşağıdaki dizin yapısına göre organize edilmiştir:

```text
GEMINI-STREAMLIT-MOVIEBOT/
├── app/
│   ├── app.py                            # Ana Streamlit uygulama dosyası
│   └── Moviebot.ipynb                    # Geliştirme ve test not defteri
├── data/
│   ├── IMDb_Top_1000_Movies_Dataset.csv  # Film veri seti (Kaynak)
│   └── intent_classification_data.csv    # Niyet sınıflandırma eğitim verisi
├── intent_classification_model/
│   ├── intent_model.pkl                  # Eğitilmiş niyet sınıflandırma modeli
│   └── train_classifier.py               # Modeli yeniden eğitmek için script
├── .env                                  # API key
├── .gitignore                            # Git göz ardı dosyası
├── README.md                             # Proje dokümantasyonu
└── requirements.txt                      # Gerekli kütüphaneler
```

## 🛠️ Kurulum
Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

1. Repoyu Klonlayın
```bash

git clone https://github.com/kullaniciadi/moviebot-rag.git
cd GEMINI-STREAMLIT-MOVIEBOT

```
2. Conda Ortamını Oluşturun
Miniconda veya Anaconda kurulu ise, proje için yeni bir sanal ortam oluşturun:

```Bash

# Python 3.10 tabanlı yeni bir ortam oluştur
conda create --name rag_env python=3.10

# Ortamı aktif et
conda activate moviebot-rag

```

3. Bağımlılıkları Yükleyin
Conda ortamı aktifken gerekli kütüphaneleri yükleyin:

```bash

pip install -r requirements.txt

```
4. Çevresel Değişkenleri Ayarlayın
Proje ana dizininde .env adında bir dosya oluşturun ve Google Gemini API anahtarınızı ekleyin:

``` bash

GOOGLE_API_KEY=senin_google_api_anahtarin_buraya

```

## ▶️ Uygulamayı Çalıştırma
Proje kök dizinindeyken, uygulamayı app klasörü içinden başlatmak için şu komutu kullanın:

```Bash

streamlit run app/app.py
```

Tarayıcınızda http://localhost:8501 adresi otomatik olarak açılacaktır.

🧠 Model Eğitimi (Opsiyonel)
Eğer niyet sınıflandırma modelini güncellemek veya yeni verilerle tekrar eğitmek isterseniz:

- data/intent_classification_data.csv dosyasını güncelleyin.

- Ana dizinde şu komutu çalıştırın:

```Bash

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

## Çalışma Videosu


