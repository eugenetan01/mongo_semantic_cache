# Setup

__1. Create python3 env and activate it__
  a. `python3 -m venv myenv`
  b. `source myenv/bin/activate`

__2. Install requirements__
  a. `pip3 install -r requirements.txt`

__3. Create a .env file and add the openai key as the following param__
  a. OPENAI_API_KEY="<key>"

__4. Create a db.coll on mongodb called "langchain_db.langchain_cache"__

__5. Create a vector search index called "vector_index" and add the following def:__
  ```
  {
    "fields": [
      {
        "numDimensions": 1024,
        "path": "embedding",
        "similarity": "cosine",
        "type": "vector"
      },
      {
        "path": "llm_string",
        "type": "filter"
      }
    ]
  }
  ```

__6. create the following .env file with corresponding vars__
  ```
  OPENAI_API_KEY="<key>"
  MONGODB_URI="mongodb+srv://<user>:<pass>@eugenemongo.ibdxz.mongodb.net/?retryWrites=true&w=majority&appName=EugeneMongo"
  ATLAS_VECTOR_SEARCH_INDEX_NAME="vector_index"
  ```

# Execution

__1. Run the in_mem_cache.py file to see the cb1 and cb2 run with tokens and cost -  meaning OpenAI LLM ran the processing__
  - show that cb3 did not call OpenAI LLM as the question was an exact match to cb1, and langchain did in mem caching to process the answer

__2. Run the semantic_cache.py file to see that cb1 may have run with some tokens cost, but cb2 and cb3 leveraged semantic caching__
  - cb1 might have incurred token cost and LLM processing
  - cb2 and cb3 had no LLM processing, 0.0 tokens processed and cost is $0 as the semantic cache processed the query

__3. Change cb2 - similar question var to "What are the ingredients to cook a tandoori naan bread" and see that it hits the LLM for processing__
  - the tuning of the accuracy of the semantic cache is done on the `score_threshold=0.8,` in the below code block:
  ```
  set_llm_cache(MongoDBAtlasSemanticCache(
      embedding=embeddings,
      connection_string=MONGODB_ATLAS_CLUSTER_URI,
      collection_name=COLLECTION_NAME,
      database_name=DATABASE_NAME,
      index_name=os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME"),
      score_threshold=0.8,
      wait_until_ready=True # Optional, waits until the cache is ready to be used
  ))
  ```
  - this determines the sensitivity of the cosine function when retrieving results from the vector db

__Notes:__
- Guide [here](https://towardsdatascience.com/maximizing-ai-efficiency-in-production-with-caching-a-cost-efficient-performance-booster-9b8afd200efd)
