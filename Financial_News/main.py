import os
from dotenv import load_dotenv
import streamlit as st
import pickle
import time
import langchain

from langchain_openai import OpenAI
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)

st.title("News Research Tool")
st.sidebar.title("News Article URLs")
main_placeholder = st.empty()

if "vectorindex_openai" not in st.session_state:
    st.session_state.vectorindex_openai = None

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")    

if process_url_clicked and any(url.strip()):
    # load data
    main_placeholder.text("Loading data is in progress...")
    url_loader = UnstructuredURLLoader(urls = urls)

    data = url_loader.load()

    # split data
    main_placeholder.text("Splitting data is in progress...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        # chunk_overlap = 200,
        separators = ["\n\n", "\n", ".", ","]
    )
    docs = text_splitter.split_documents(data)

    # create embedding 
    main_placeholder.text("Embedding is in progress...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
    # vector_store = Chroma.from_documents(docs, embeddings)
    st.session_state.vectorindex_openai = FAISS.from_documents(docs, embeddings)

# else:
#     st.sidebar.error("All URL fields are mandatory! Please fill in both fields.")

query = main_placeholder.text_input("Question: ")
if query:
    # main_placeholder.text("Thinking...")
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=st.session_state.vectorindex_openai.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)
    
    st.header("Answer: ")
    st.write(result["answer"])

    # display sources if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources: ")
        sources_list = sources.split("\n")
        for source in sources_list:
            st.write(source)
            

