import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import joblib
import os

# 1. Ayarlar ve API Anahtarları
st.set_page_config(page_title="MovieBot RAG", layout="wide")
load_dotenv()

st.title("🎬 AI Destekli Film Asistanı")

# --- 2. ÖNBELLEKLEME (PERFORMANS İÇİN) ---

@st.cache_resource
def load_intent_model():
    """Eğitilmiş intent sınıflandırma modelini yükler."""
    model_path = 'models/intent_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def initialize_vectorstore():
    """Film verisetini yükler ve vektör veritabanını hazırlar."""
    if not os.path.exists("IMDB_Top_1000_Movies_Dataset.csv"):
        st.error("Veri seti dosyası (csv) bulunamadı!")
        return None

    loader = CSVLoader("IMDB_Top_1000_Movies_Dataset.csv", encoding='utf-8')
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_batch_size=100)
    vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
    
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Kaynakları yükle
classifier = load_intent_model()
retriever = initialize_vectorstore()

# --- 3. SABİT GEMINI MODELİ TANIMI ---
# Model seçimi kaldırıldı, doğrudan Gemini tanımlanıyor.
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # 2.5 henüz genel kullanıma açık değil, 1.5 en kararlı sürüm
    temperature=0.3,
    max_tokens=500
)

# --- 4. RAG PROMPT TASARIMI ---
system_prompt = (
    "Sen yardımsever bir film öneri asistanısın. Aşağıdaki film veritabanı bağlamını (context) kullanarak kullanıcının sorularını yanıtla."
    "\n\n"
    "Kurallar:"
    "1. Sadece verilen bağlamdaki (context) filmleri öner."
    "2. İlgili yerlerde IMDB puanı, yıl ve oyuncu bilgilerini belirt."
    "3. Eğer bağlamda cevap yoksa, dürüstçe 'Veri setimde bu bilgi yok' de."
    "\n\n"
    "Context:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

# --- 5. CHAT ARAYÜZÜ VE AKIŞ ---

# Yan Menü (Sadece temizleme butonu kaldı)
with st.sidebar:
    st.header("⚙️ İşlemler")
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()
    
    # Debug: Modelin yüklü olup olmadığını göster
    if classifier:
        st.success("✅ Intent Modeli Aktif")
    else:
        st.warning("⚠️ Intent Modeli Yüklenemedi")

# Sohbet Geçmişini Başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# KULLANICI GİRDİSİ
if query := st.chat_input("Film sorun veya sohbet edin..."):
    
    # 1. Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    response_text = ""
    intent = "MOVIE_QUERY" # Varsayılan

    # 2. INTENT (NİYET) TAHMİNİ
    if classifier:
        intent = classifier.predict([query])[0]
        # İsteğe bağlı: Niyeti debug için konsola veya sidebara yazdırabilirsin
        # print(f"Tahmin edilen niyet: {intent}")

    # 3. NİYETE GÖRE CEVAPLAMA MANTIĞI
    with st.chat_message("assistant"):
        
        # A) Sohbet / Selamlaşma (LLM Harcamaz)
        if intent == "GREETING":
            response_text = "Merhaba! Size filmler hakkında nasıl yardımcı olabilirim? 🎬"
            st.write(response_text)
            
        elif intent == "GOODBYE":
            response_text = "Görüşmek üzere! İyi seyirler dilerim."
            st.write(response_text)
            
        elif intent == "CHITCHAT":
            response_text = "Ben sadece filmlerden anlayan bir asistanım. Bana favori türünü sorabilirsin!"
            st.write(response_text)
            
        elif intent == "REJECT":
            response_text = "Anladım, başka bir öneri ister misin?"
            st.write(response_text)
            
        elif intent == "OTHER":
            response_text = "Üzgünüm, siyaset, hava durumu veya yemek tarifleri alanım dışı. Sadece sinema konuşalım! 🍿"
            st.write(response_text)
            
        # B) Film Sorusu (RAG Devreye Girer)
        else: # MOVIE_QUERY veya Tanımsız
            with st.spinner("Veritabanı taranıyor..."):
                if retriever:
                    question_answering_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(retriever, question_answering_chain)
                    
                    response = rag_chain.invoke({"input": query})
                    response_text = response["answer"]
                    st.write(response_text)
                else:
                    response_text = "Veritabanı bağlantısı kurulamadı."
                    st.error(response_text)

    # 4. Asistan cevabını geçmişe kaydet
    st.session_state.messages.append({"role": "assistant", "content": response_text})