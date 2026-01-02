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


load_dotenv()

st.title("Moviebot")

loader = CSVLoader("IMDB_Top_1000_Movies_Dataset.csv", encoding ='utf-8')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_batch_size=100)
vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5 })

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  
    temperature=0.3,  
    max_tokens=500,
)

query = st.text_input("Enter your question about movies:")
prompt = query

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

if query:
    question_answering_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answering_chain)
    response = rag_chain.invoke({"input": query})

    st.write(response["answer"])