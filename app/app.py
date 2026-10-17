# --- 0. GEREKLİ KÜTÜPHANELERİN İÇE AKTARIMI ---
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
import joblib
import os
import re
import pandas as pd


# --- 1. STREAMLIT SAYFA AYARLARI ---

st.set_page_config(page_title="MovieBot RAG", layout="wide")
load_dotenv()

st.title("🎬 Moviebot")


@st.cache_resource
def load_intent_model():
    """Eğitilmiş intent sınıflandırma modelini yükler."""
    model_path = 'intent_classification_model/intent_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# --- 2. VERİ SETİNİ YÜKLEME + VEKTÖR DEPOSU + FİLTRELİ ARAMA ---

DATA_PATH = "data/IMDb_Top_1000_Movies_Dataset.csv"
PERSIST_DIR = "chroma_db_v2"  # Metin formatı değiştiği için yeni klasör
# Çok dilli yerel model: Türkçe sorular İngilizce film açıklamalarıyla eşleşebilir.
# İlk çalıştırmada (~470 MB) bir kez indirilir, Gemini kotası harcamaz.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_CONTEXT_MOVIES = 15

# Sorudaki kelime -> veri setindeki tür (İngilizce + Türkçe)
GENRE_KEYWORDS = {
    "Comedy": ["comedy", "comedies", "funny", "komedi", "komik"],
    "Horror": ["horror", "scary", "korku"],
    "Drama": ["drama", "dram"],
    "Action": ["action", "aksiyon"],
    "Sci-Fi": ["sci-fi", "science fiction", "scifi", "bilim kurgu", "bilimkurgu"],
    "Animation": ["animation", "animated", "cartoon", "animasyon", "çizgi film"],
    "Romance": ["romance", "romantic", "love story", "romantik", "aşk"],
    "War": ["war film", "war movie", "war movies", "war films", "savaş"],
    "Crime": ["crime", "suç"],
    "Thriller": ["thriller", "gerilim"],
    "Mystery": ["mystery", "gizem"],
    "Fantasy": ["fantasy", "fantastik"],
    "Adventure": ["adventure", "macera"],
    "Family": ["family", "aile"],
    "Biography": ["biography", "biopic", "biyografi"],
    "History": ["historical", "tarihi"],
    "Musical": ["musical", "müzikal"],
    "Western": ["western", "kovboy"],
    "Sport": ["sport", "sports", "spor"],
}


@st.cache_resource
def load_movies():
    """CSV'yi okur ve her film için temiz bir metin hazırlar."""
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df.drop_duplicates(subset=["Movie_Name", "Movie_Year"]).reset_index(drop=True)
    for col in ["Movie_Name", "Movie_Cast", "Movie_Genre", "Movie_Description", "Movie_Certificate", "Movie_Runtime"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["text"] = (
        "Title: " + df["Movie_Name"]
        + "\nYear: " + df["Movie_Year"].astype(str)
        + "\nIMDb Rating: " + df["Movie_Rating"].astype(str)
        + "\nGenre: " + df["Movie_Genre"]
        + "\nCast: " + df["Movie_Cast"]
        + "\nRuntime: " + df["Movie_Runtime"]
        + "\nCertificate: " + df["Movie_Certificate"]
        + "\nDescription: " + df["Movie_Description"]
    )
    # Veri setindeki tüm oyuncu adları (eşleştirme için, küçük harf)
    actors = set()
    for cast in df["Movie_Cast"]:
        for name in cast.split(","):
            name = name.strip()
            if len(name.split()) >= 2:  # Tek kelimelik adlar yanlış eşleşmeye açık
                actors.add(name.lower())
    return df, actors


def to_document(row):
    return Document(
        page_content=row["text"],
        metadata={"title": row["Movie_Name"], "year": int(row["Movie_Year"]), "rating": float(row["Movie_Rating"])},
    )


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def initialize_vectorstore():
    """Vektör deposunu diskten yükler; yoksa bir kez oluşturup kaydeder."""
    embeddings = get_embeddings()

    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    loaded = load_movies()
    if loaded is None:
        st.error("Dataset file not found.")
        return None
    df, _ = loaded
    docs = [to_document(row) for _, row in df.iterrows()]

    with st.spinner("Film veritabanı ilk kez hazırlanıyor (bir kerelik, birkaç dakika sürebilir)..."):
        return Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=PERSIST_DIR)


def find_filters(query, actors):
    """Sorudaki oyuncu, tür ve yıl/on yıl ifadelerini bulur."""
    q = query.lower()
    found_actors = [a for a in actors if a in q]
    found_genres = [g for g, words in GENRE_KEYWORDS.items()
                    if any(re.search(r"\b" + re.escape(w), q) for w in words)]

    year_range = None
    decade = re.search(r"\b(?:19|20)?(\d)0\s*(?:'|’)?\s*(?:s\b|lar|ler)", q)
    year = re.search(r"\b(19\d{2}|20\d{2})\b", q)
    if decade:
        d = int(decade.group(1))
        full = re.search(r"\b(19|20)\d0", decade.group(0))
        century = int(full.group(1)) * 100 if full else (1900 if d >= 2 else 2000)
        start = century + d * 10
        year_range = (start, start + 9)
    elif year:
        y = int(year.group(1))
        year_range = (y, y)
    return found_actors, found_genres, year_range


def retrieve_movies(query, vector_store, movies):
    """Önce kesin filtre (oyuncu/tür/yıl), sonuç yoksa anlamsal arama."""
    df, actors = movies
    found_actors, found_genres, year_range = find_filters(query, actors)

    def apply_filters(actors_, genres_, years_):
        mask = pd.Series(True, index=df.index)
        for a in actors_:
            mask &= df["Movie_Cast"].str.lower().str.contains(a, regex=False)
        for g in genres_:
            mask &= df["Movie_Genre"].str.contains(g, regex=False)
        if years_:
            mask &= df["Movie_Year"].between(*years_)
        return df[mask]

    # Tam eşleşme yoksa filtreleri sırayla gevşet: önce yıl, sonra tür
    attempts = [
        (found_actors, found_genres, year_range),
        (found_actors, found_genres, None),
        (found_actors, [], year_range),
        (found_actors, [], None),
    ]
    for actors_, genres_, years_ in attempts:
        if not (actors_ or genres_ or years_):
            continue
        matches = apply_filters(actors_, genres_, years_)
        if len(matches) == 0:
            continue

        if len(matches) > MAX_CONTEXT_MOVIES:
            # Çok sonuç varsa soruya anlamca en yakın olanları seç
            hits = vector_store.similarity_search(query, k=200)
            allowed = set(matches["Movie_Name"])
            ranked = [d.metadata["title"] for d in hits if d.metadata.get("title") in allowed]
            order = ranked + [t for t in matches.sort_values("Movie_Rating", ascending=False)["Movie_Name"] if t not in ranked]
            matches = matches.set_index("Movie_Name").loc[order[:MAX_CONTEXT_MOVIES]].reset_index()

        if len(matches) > 0:
            return [to_document(row) for _, row in matches.iterrows()]

    return vector_store.similarity_search(query, k=8)


# Kaynakları yükle

classifier = load_intent_model()
movies = load_movies()
vector_store = initialize_vectorstore()

# --- 3. GEMINI MODELİ TANIMI ---

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    max_tokens=500
)

# --- 4. RAG PROMPT TASARIMI ---
system_prompt = (
    "You are a helpful movie recommendation assistant. Use the following movie database context to answer the user's question."    "\n\n"
    "\n\n"
    "Rules:"
    "1. Only recommend movies from the provided context. "
    "2. Mention IMDB ratings and year where relevant. "
    "3. If you don't find suitable movies in the context, say so honestly. "
    "4. A cast member voicing an animated character counts as starring in it. "
    "5. Answer in the same language as the user's question."
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

# Yan Menü (Temizleme butonu)
with st.sidebar:
    st.header("⚙️ Functions")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Debug: Modelin yüklü olup olmadığını göster
    if classifier:
        st.success("✅ Intent Model Active")
    else:
        st.warning("⚠️ Intent Model Inactivate.")

# Sohbet Geçmişini Başlat

if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana bas

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# KULLANICI GİRDİSİ

if query := st.chat_input("Ask me movies..."):
    
    # 1. Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    response_text = ""
    intent = "MOVIE_QUERY" # Varsayılan

    # 2. INTENT (NİYET) TAHMİNİ
    if classifier:
        intent = classifier.predict([query])[0]

    # 3. NİYETE GÖRE CEVAPLAMA MANTIĞI
    with st.chat_message("assistant"):
        
        # A) Sohbet / Selamlaşma / Red / Vedalaşma (LLM Harcamaz)
        if intent == "GREETING":
            response_text = "Hello! How can i help you? 🎬"
            st.write(response_text)
            
        elif intent == "GOODBYE":
            response_text = "See you later! Enjoy your movies! 🍿"
            st.write(response_text)
            
        elif intent == "CHITCHAT":
            response_text = "I am here to chat about movies! What would you like to know? 🎥"
            st.write(response_text)
            
        elif intent == "REJECT":
            response_text = "I understand. If you have any movie questions later, feel free to ask! 🎬"
            st.write(response_text)
            
        elif intent == "OTHER":
            response_text = "I'm sorry, but I can't discuss politics, weather, or recipes etc. Let's keep the conversation focused on movies! 🍿"
            st.write(response_text)
            
        # B) Film Sorusu (RAG Devreye Girer)
        else: # MOVIE_QUERY veya Tanımsız
            with st.spinner("Database searching..."):
                if vector_store and movies:
                    question_answering_chain = create_stuff_documents_chain(llm, prompt_template)
                    context_docs = retrieve_movies(query, vector_store, movies)

                    response_text = question_answering_chain.invoke({"input": query, "context": context_docs})
                    st.write(response_text)
                else:
                    response_text = "Database not connected."
                    st.error(response_text)

    # 4. Asistan cevabını geçmişe kaydet
    st.session_state.messages.append({"role": "assistant", "content": response_text})