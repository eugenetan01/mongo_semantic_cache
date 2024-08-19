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
# Initialize MongoDB python client

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model="gpt-3.5-turbo-instruct")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_URI')# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

COLLECTION_NAME="langchain_cache"
DATABASE_NAME="langchain_db"
question="What are the ingredients to cook a pizza?"

set_llm_cache(MongoDBAtlasSemanticCache(
    embedding=embeddings,
    connection_string=MONGODB_ATLAS_CLUSTER_URI,
    collection_name=COLLECTION_NAME,
    database_name=DATABASE_NAME,
    index_name=os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME"),
    score_threshold=0.8,
    wait_until_ready=True # Optional, waits until the cache is ready to be used
))

question="How to make a pizza ?"
similar_question="What are the ingredients to cook a pizza"

with get_openai_callback() as cb:
    start = time.time()
    result = llm(question)
    end = time.time()
    print(result)
    print("--- cb")
    print(str(cb) + f"({end - start:.2f} seconds)")
time.sleep(5)

with get_openai_callback() as cb2:
     start = time.time()
     print("##### CB2 #######")
     print(similar_question)
     result2 = llm(similar_question)
     end = time.time()
     print(result2)
     print("--- cb2")
     print(str(cb2) + f"({end - start:.2f} seconds)")
time.sleep(5)

with get_openai_callback() as cb3:
     start = time.time()
     result3 = llm(question)
     end = time.time()
     print(result3)
     print("--- cb3")
     print(str(cb3) + f"({end - start:.2f} seconds)")
