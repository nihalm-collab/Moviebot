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
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

st.title("Movie Dictionary with Google Generative AI")

loader = CSVLoader("IMDB_Top_1000_Movies_Dataset.csv", encoding="utf-8")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5 })

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.3,  
    max_tokens=500,
)

query = st.text_input("Enter your question about movies:")
prompt = query

intent_system_prompt = (
    "You are a text classifier. Analyze the user's input and classify it into exactly one of these categories:\n"
    "1. MOVIE_QUERY: If the user is asking for movie recommendations, details about a specific movie, plot, cast, or ratings.\n"
    "2. CHITCHAT: If the user is greeting (hi, hello) or asking general questions like 'how are you'.\n"
    "3. OTHER: Anything else unrelated to movies or greetings.\n\n"
    "Return ONLY the category name (MOVIE_QUERY, CHITCHAT, or OTHER). Do not write anything else."
)

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", intent_system_prompt),
    ("user", "{input}")
])

intent_chain = intent_prompt | llm | StrOutputParser()

system_prompt = (
    "You are a helpful movie recommendation assistant. Use the following movie database context to answer the user's question."
    "Only recommend movies from the provided context"
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

question_answering_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answering_chain)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant. Answer the user politely."),
    ("user", "{input}")
])

simple_chat_chain = chat_prompt | llm | StrOutputParser()

if query:
    with st.spinner("Analyzing intent..."):
        intent = intent_chain.invoke({"input": query}).strip()
        st.caption(f"Detected Intent: {intent}")

    if intent == "MOVIE_QUERY":
        with st.spinner("Searching movie database..."):
            response = rag_chain.invoke({"input": query})
            st.write(response["answer"])
            
    elif intent == "CHITCHAT":
        response = simple_chat_chain.invoke({"input": query})
        st.write(response)
        
    else: 
        st.warning("I can only answer questions about movies or have a simple chat. Please ask about a movie!")