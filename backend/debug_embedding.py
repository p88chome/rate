import os
import sys
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

# Force load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
loaded = load_dotenv(env_path)

print("=== DEBUG EMBEDDING START ===")
print(f"Loading .env from: {env_path}")
print(f"Env loaded: {loaded}")
print(f"AZURE_OPENAI_API_KEY: {'[HIDDEN]' if os.getenv('AZURE_OPENAI_API_KEY') else 'None'}")
print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"AZURE_OPENAI_EMBEDDING_DEPLOYMENT: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')}")
print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION')}")
print("==================")

try:
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    print("Attempting to embed query 'test'...")
    res = embeddings.embed_query("test")
    print(f"Success! Embedding length: {len(res)}")
    print("First 5 values:", res[:5])
except Exception as e:
    print(f"FATAL ERROR: {e}")
