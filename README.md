# 🎬 MovieBot RAG

**MovieBot**, film severler için geliştirilmiş, kaggle'dan https://www.kaggle.com/datasets/hrishabhtiwari/imdb-top-1000-movies-dataset bağlantısı üzerinden indirilen **IMDb_Top_1000_Movies_Dataset.csv** veri setini ve **Google Gemini** yapay zekasını kullanan hibrit bir sohbet asistanıdır.

Bu proje, maliyet ve performansı optimize etmek için **Intent Sınıflandırma** (Niyet Analizi) ile **RAG (Retrieval-Augmented Generation)** mimarisini bir arada kullanır. Basit sohbetler yerel bir modelle, film soruları ise Gemini AI ile yanıtlanır. Film aramak için gereken vektörler de yerel bir modelle hesaplanır; böylece Gemini kotası yalnızca cevap üretirken harcanır.

## 🚀 Özellikler

* **🧠 Hibrit Yapı:**
    * **Niyet Analizi:** Kullanıcının amacını (Selamlaşma, Film Sorusu vb.) yerel bir modelle tespit eder.
    * **RAG Motoru:** Film soruları için veri setinden bağlam çeker.
* **🎯 Filtreli Arama:** Soruda oyuncu adı, tür (İngilizce/Türkçe) veya yıl/on yıl geçiyorsa önce tabloda kesin filtre uygular (ör. *"comedy movie with Tom Hanks"*, *"90'lardan korku filmi"*). Tam eşleşme yoksa filtreleri sırayla gevşetir.
* **🌍 Çok Dilli:** Türkçe veya İngilizce sorabilirsiniz; bot sorunun dilinde cevap verir.
* **📂 Modüler Yapı:** Veriler, model eğitimi ve uygulama mantığı ayrı klasörlerde organize edilmiştir.
* **🔍 Vektör Arama:** Filtre uygulanamayan sorularda (*"Inception'a benzer bir film"*) `ChromaDB` ile anlamsal arama yapar. Vektörler yerel `paraphrase-multilingual-MiniLM-L12-v2` modeliyle hesaplanır ve `chroma_db_v2/` klasörüne kaydedilir; yalnızca ilk açılışta oluşturulur.
* **🤖 Google Gemini:** Doğal dil işleme ve cevap üretimi için `gemini-2.5-flash-lite` modelini kullanır.

## 📂 Proje Yapısı

Proje dosyaları aşağıdaki dizin yapısına göre organize edilmiştir:

```text
Moviebot/
├── app/                          
│   └── app.py                     # Ana Streamlit uygulama dosyası
├── data/
│   ├── IMDb_Top_1000_Movies_Dataset.csv  # Film veri seti (Kaynak)
│   └── intent_classification_dataset.csv # Niyet sınıflandırma eğitim verisi
├── intent_classification_model/
│   ├── intent_model.pkl                  # Eğitilmiş niyet sınıflandırma modeli
│   └── train_classifier.py               # Modeli yeniden eğitmek için script
├── chroma_db_v2/                         # Vektör veritabanı (ilk açılışta otomatik oluşur, git'e eklenmez)
├── .env                                  # API key (git'e eklenmez)
├── .gitignore                            # Git göz ardı dosyası
├── README.md                             # Proje dokümantasyonu
└── requirements.txt                      # Gerekli kütüphaneler
```

## 🛠️ Kurulum
Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

1. Repoyu Klonlayın
```bash

git clone https://github.com/nihalm-collab/Moviebot.git
cd Moviebot

```
2. Conda Ortamını Oluşturun
Miniconda veya Anaconda kurulu ise, proje için yeni bir sanal ortam oluşturun:

```Bash

# Python 3.10 tabanlı yeni bir ortam oluştur
conda create --name rag_env python=3.10

# Ortamı aktif et
conda activate rag_env

```

3. Bağımlılıkları Yükleyin
Conda ortamı aktifken gerekli kütüphaneleri yükleyin:

```bash

pip install -r requirements.txt

```
4. Çevresel Değişkenleri Ayarlayın
Proje ana dizininde .env adında bir dosya oluşturun ve [Google AI Studio](https://aistudio.google.com/apikey)'dan aldığınız Gemini API anahtarınızı ekleyin:

``` bash

GOOGLE_API_KEY=senin_google_api_anahtarin_buraya

```

> ⚠️ `.env` dosyası `.gitignore` içindedir; API anahtarınızı asla GitHub'a yüklemeyin. Yüklenen anahtarlar Google tarafından tespit edilip iptal edilir.

## ▶️ Uygulamayı Çalıştırma
Proje kök dizinindeyken, uygulamayı app klasörü içinden başlatmak için şu komutu kullanın:

```Bash

streamlit run app/app.py
```

Tarayıcınızda http://localhost:8501 adresi otomatik olarak açılacaktır.

**İlk açılış:** Embedding modeli (~470 MB) bir kez indirilir ve 1000 filmin vektörleri hesaplanıp `chroma_db_v2/` klasörüne kaydedilir. Bu birkaç dakika sürebilir; sonraki açılışlar hızlıdır. Veri setini değiştirirseniz bu klasörü silin, bir sonraki açılışta yeniden oluşturulur.

## 🧠 Model Eğitimi (Opsiyonel)
Eğer niyet sınıflandırma modelini güncellemek veya yeni verilerle tekrar eğitmek isterseniz:

- data/intent_classification_dataset.csv dosyasını güncelleyin.

- Ana dizinde şu komutu çalıştırın:

```Bash

python intent_classification_model/train_classifier.py

```

## 🛠️ Kullanılan Teknolojiler

- Python 3.10

- Streamlit (Arayüz)

- LangChain (RAG Orkestrasyonu)

- Google Gemini API (cevap üretimi)

- Sentence-Transformers / HuggingFace (yerel embedding)

- Pandas (filtreli arama)

- ChromaDB (Vektör Veritabanı)

- Scikit-Learn (Niyet Sınıflandırma)

- Miniconda (Ortam Yönetimi)

## Çalışma Videosu

https://github.com/user-attachments/assets/6eff5bd4-ec26-4d9e-96f0-c1c9ba09ec3b