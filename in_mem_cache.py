import langchain
import time
import os
from langchain.llms import OpenAI
from langchain.cache import InMemoryCache
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
# Initialize MongoDB python client

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

langchain.llm_cache = InMemoryCache()

llm = OpenAI(model="gpt-3.5-turbo-instruct")
question = "What are the ingredients to cook a pizza?"

with get_openai_callback() as cb:
    start = time.time()
    result = llm(question)
    end = time.time()
    print(result)
    print("--- cb")
    print(str(cb) + f"({end - start:.2f} seconds)")

with get_openai_callback() as cb2:
    start = time.time()
    result2 = llm("What are the ingredients to cook pizzas?")
    end = time.time()
    print(result2)
    print("--- cb2")
    print(str(cb2) + f"({end - start:.2f} seconds)")


with get_openai_callback() as cb3:
    start = time.time()
    result3 = llm(question)
    end = time.time()
    print(result3)
    print("--- cb2")
    print(str(cb3) + f"({end - start:.2f} seconds)")
