import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import pandas as pd

load_dotenv()

st.set_page_config(page_title="MovieBot RAG", layout="wide")

st.title("🎬 IMDB Film Asistanı")

@st.cache_resource
def load_intent_model():
    # Eğittiğimiz modeli yükle
    if os.path.exists('models/intent_model.pkl'):
        return joblib.load('models/intent_model.pkl')
    return None

classifier = load_intent_model()

loader = CSVLoader("IMDB_Top_1000_Movies_Dataset.csv", encoding ='utf-8')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

structured_docs = []
for doc in data:
    structured_docs.append(doc)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_batch_size=100)
vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5 })

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.3,  
    max_tokens=500,
)

query = st.text_input("Enter your question about movies:")
prompt = query

system_prompt = (
    "You are a helpful movie recommendation assistant. Use the following movie database context to answer the user's question."
    "Only recommend movies from the provided context"
    "\n- Use EXACT values for Movie_Rating, Movie_Runtime, and other numerical data from the context"
    "Mention IMDB ratings and year where relevant."
    "If you don't find suitable movies in the context, say so honestly."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

if query := st.chat_input("Mesajınızı yazın..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # 1. NİYET ANALİZİ (INTENT CLASSIFICATION)
    intent = "MOVIE_QUERY" # Model yoksa varsayılan
    if classifier:
        intent = classifier.predict([query])[0]
    
    # Debug için yan tarafa yazdırabilirsiniz (Hoca görsün diye)
    st.sidebar.markdown(f"**Tespit Edilen Niyet:** `{intent}`")

    response_text = ""

    # 2. INTENT'E GÖRE AKSİYON
    if intent == "GREETING":
        response_text = "Merhaba! Ben bir film uzmanıyım. Sana nasıl yardımcı olabilirim?"
        
    elif intent == "GOODBYE":
        response_text = "Görüşmek üzere! İyi seyirler."
        
    elif intent == "CHITCHAT":
        response_text = "Ben bir yapay zeka asistanıyım, sadece filmlerden anlarım! 🎬"
        
    elif intent == "REJECT":
        response_text = "Pekala, başka bir konuda yardımcı olmamı ister misin?"
        
    elif intent == "OTHER":
        response_text = "Üzgünüm, şu an sadece filmler hakkında konuşabiliyorum. Siyaset veya yemek tarifleri alanım dışı. 😊"
        
    elif intent == "MOVIE_QUERY":
        # --- BURADA MEVCUT RAG ZİNCİRİNİZ ÇALIŞACAK ---
        with st.chat_message("assistant"):
            with st.spinner(f"{model_choice} veritabanını tarıyor..."):
                if retriever:
                    question_answering_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(retriever, question_answering_chain)
                    response = rag_chain.invoke({"input": query})
                    response_text = response["answer"]
                else:
                    response_text = "Veritabanı bağlantısında sorun var."

    # Cevabı Yazdır (Eğer RAG değilse yukarıda atanmıştı, RAG ise zaten yazıldı ama geçmişe eklemek lazım)
    if intent != "MOVIE_QUERY":
        with st.chat_message("assistant"):
            st.write(response_text)
            
    st.session_state.messages.append({"role": "assistant", "content": response_text})