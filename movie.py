import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Intent tanımları
intents = {
    "greeting": "Selamlama, merhaba deme, hoş geldin deme gibi",
    "goodbye": "Vedalaşma, görüşürüz, hoşça kal gibi",
    "movie_recommendation": "Film önerisi isteme, hangi film izlemeliyim, öneri",
    "movie_info": "Belirli bir film hakkında bilgi, oyuncular, konu, yönetmen",
    "rating_query": "IMDB puanı, değerlendirme, kaç puan almış",
    "chitchat": "Nasılsın, ne yapıyorsun, hava nasıl gibi genel sorular",
    "out_of_scope": "Film dışı konular, yemek tarifi, spor, politika"
}

def generate_intent_data(intent_name, description, count=150):
    """Gemini ile intent verisi üret"""
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    "{intent_name}" kategorisi için {count} adet Türkçe örnek cümle üret.
    Kategori açıklaması: {description}
    
    Kurallar:
    - Her cümle farklı olmalı
    - Doğal konuşma dili kullan
    - Kısa ve uzun cümleler karışık olsun
    - Sadece cümleleri listele, numara veya açıklama ekleme
    - Her satıra bir cümle
    
    Örnek format:
    Merhaba
    Selam nasılsın
    Hey
    """
    
    try:
        response = model.generate_content(prompt)
        sentences = [s.strip() for s in response.text.split('\n') if s.strip()]
        return [(intent_name, sent) for sent in sentences[:count]]
    except Exception as e:
        print(f"Hata {intent_name}: {e}")
        return []

# Veri üretimi
all_data = []
for intent, description in intents.items():
    print(f"📝 {intent} verisi üretiliyor...")
    data = generate_intent_data(intent, description, 150)
    all_data.extend(data)
    time.sleep(2)  # API rate limit

# DataFrame oluştur
df = pd.DataFrame(all_data, columns=['intent', 'text'])

# Karıştır ve kaydet
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/intent_dataset.csv', index=False, encoding='utf-8')

print(f"\n✅ Toplam {len(df)} satır veri oluşturuldu!")
print(df['intent'].value_counts())