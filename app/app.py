import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from dotenv import load_dotenv
import joblib
import os


st.set_page_config(page_title="MovieBot RAG", layout="wide")
load_dotenv()

st.title("🎬 Moviebot")



@st.cache_resource
def load_intent_model():
    """Eğitilmiş intent sınıflandırma modelini yükler."""
    model_path = 'intent_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def initialize_vectorstore():
    """Uploads and initializes the vector store from the CSV dataset."""
    if not os.path.exists("IMDB_Top_1000_Movies_Dataset.csv"):
        st.error("Dataset file not found.")
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

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    max_tokens=500
)

# --- 4. RAG PROMPT TASARIMI ---
system_prompt = (
    "You are a helpful movie recommendation assistant. Use the following movie database context to answer the user's question."    "\n\n"
    "\n\n"
    "Rules:"
    "1. Only recommend movies from the provided context"
    "2. Mention IMDB ratings and year where relevant."
    "3. If you don't find suitable movies in the context, say so honestly."
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
    st.header("⚙️ Functions")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Debug: Modelin yüklü olup olmadığını göster
    if classifier:
        st.success("✅ Intent Model Active")
    else:
        st.warning("⚠️ Intent Model not uploaded.")

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
        # İsteğe bağlı: Niyeti debug için konsola veya sidebara yazdırabilirsin
        # print(f"Tahmin edilen niyet: {intent}")

    # 3. NİYETE GÖRE CEVAPLAMA MANTIĞI
    with st.chat_message("assistant"):
        
        # A) Sohbet / Selamlaşma (LLM Harcamaz)
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
            response_text = "I'm sorry, but I can't discuss politics, weather, or recipes. Let's keep the conversation focused on movies! 🍿"
            st.write(response_text)
            
        # B) Film Sorusu (RAG Devreye Girer)
        else: # MOVIE_QUERY veya Tanımsız
            with st.spinner("Database searching..."):
                if retriever:
                    question_answering_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(retriever, question_answering_chain)
                    
                    response = rag_chain.invoke({"input": query})
                    response_text = response["answer"]
                    st.write(response_text)
                else:
                    response_text = "Database not connected."
                    st.error(response_text)

    # 4. Asistan cevabını geçmişe kaydet
    st.session_state.messages.append({"role": "assistant", "content": response_text})