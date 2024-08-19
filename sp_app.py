import streamlit as st
from pymongo import MongoClient
from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain_mongodb.cache import MongoDBCache
from langchain_core.globals import set_llm_cache
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
import time
from langchain_mongodb.cache import MongoDBAtlasSemanticCache
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize MongoDB python client
MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_URI')
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

# Set up the collection and database names
COLLECTION_NAME = "langchain_cache"
DATABASE_NAME = "langchain_db"

# Initialize OpenAI and embeddings
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model="gpt-3.5-turbo-instruct", timeout=30)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

# Configure the MongoDB cache
set_llm_cache(MongoDBAtlasSemanticCache(
    embedding=embeddings,
    connection_string=MONGODB_ATLAS_CLUSTER_URI,
    collection_name=COLLECTION_NAME,
    database_name=DATABASE_NAME,
    index_name=os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME"),
    score_threshold=0.8,
    wait_until_ready=True  # Optional, waits until the cache is ready to be used
))

# Streamlit app
st.title("LangChain Cache Example")

# Get user input for the question
question = st.text_input("Enter your question:", value="How to make a pizza?")

# Run the LLM with caching and display the result
if st.button("Get Answer"):
    with get_openai_callback() as cb:
        start = time.time()
        result = llm(question)
        end = time.time()

        st.subheader("Answer")
        st.write(result)

        st.subheader("Callback Information")
        st.write(str(cb))
        st.write(f"Execution Time: {end - start:.2f} seconds")
