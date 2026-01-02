# 🎬 MovieBot: RAG & Intent Classification

Bu proje, **LangChain**, **Google Gemini** ve **Streamlit** kullanılarak geliştirilmiş akıllı bir film tavsiye asistanıdır. Kullanıcı sorularını önce bir **Niyet Sınıflandırma (Intent Classification)** modelinden geçirir, eğer soru film veritabanıyla ilgiliyse **RAG (Retrieval-Augmented Generation)** tekniğini kullanarak cevap verir.

## 🎥 Ekran Kaydı / Demo

Uygulamanın çalışır haldeki ekran kaydına aşağıdaki bağlantıdan ulaşabilirsiniz:

[▶️ Demo Videosunu İzle](blob:http://localhost:8501/f6a1f588-4a09-428e-aae2-10e0fac473c2)

> **Not:** Yukarıdaki bağlantı yerel bir önizleme (blob) linkidir. GitHub'da videonun herkes tarafından görüntülenebilmesi için videoyu repo içerisine yükleyip relative path (örn: `docs/demo.mp4`) verebilir veya YouTube/Vimeo linki ekleyebilirsiniz.

## ✨ Özellikler

* **Niyet Analizi (Intent Classification):** Kullanıcının girdisini analiz eder (Selamlaşma, Sohbet, Film Sorusu vb.) ve LLM maliyetini düşürmek için gereksiz sorguları filtreler.
* **RAG Mimarisi:** `IMDb_Top_1000.csv` veri setini kullanarak, sadece veritabanındaki gerçek verilere dayalı cevaplar üretir.
* **Vektör Arama:** ChromaDB ve Google Generative AI Embeddings (`models/text-embedding-004`) kullanır.
* **LLM Entegrasyonu:** Google `gemini-2.5-flash-lite` modeli ile hızlı ve doğal cevaplar sunar.
* **Kullanıcı Dostu Arayüz:** Streamlit ile geliştirilmiş modern bir sohbet arayüzü.

## 📂 Proje Yapısı

```text
📁 MovieBot
├── 📄 .env                        # API anahtarları (Google API Key)
├── 📄 .gitignore                  # Git tarafından izlenmeyecek dosyalar
├── 📄 app2.py                     # Ana Streamlit uygulama dosyası
├── 📄 IMDb_Top_1000_Movies...csv  # Film veri seti
├── 📄 intent_classification...csv # Niyet sınıflandırma eğitim verisi
├── 📄 intent_model.pkl            # Eğitilmiş niyet sınıflandırma modeli
├── 📄 requirements.txt            # Gerekli Python kütüphaneleri
├── 📄 train_classifier.py         # Niyet modelini eğiten script
└── 📄 README.md                   # Proje dökümantasyonu
