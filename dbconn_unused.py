from pymongo import MongoClient
from dotenv import load_dotenv
import os
# Initialize MongoDB python client

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = MongoClient(os.getenv("MONGODB_URI"))

def get_ingest_collection():
    DB_NAME = os.getenv("DB")
    COLLECTION_NAME = os.getenv("COLL")
    ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")
    collection = client[DB_NAME][COLLECTION_NAME]
    return collection
