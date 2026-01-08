import os
import sys
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Force load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
loaded = load_dotenv(env_path)

print("=== DEBUG START ===")
print(f"Loading .env from: {env_path}")
print(f"Env loaded: {loaded}")
print(f"AZURE_OPENAI_API_KEY: {'[HIDDEN] - Length ' + str(len(os.getenv('AZURE_OPENAI_API_KEY'))) if os.getenv('AZURE_OPENAI_API_KEY') else 'None'}")
print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"AZURE_OPENAI_CHAT_DEPLOYMENT: {os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT')}")
print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION')}")
print("==================")

try:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )
    print("Attempting to invoke LLM locally...")
    res = llm.invoke("Hello")
    print(f"Success! LLM Response: {res.content}")

    print("\n=== DEBUGGING AGENT ===")
    import pandas as pd
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
    print("Attempting to invoke Agent...")
    res_agent = agent.invoke("how many rows are there?")
    print(f"Success! Agent Response: {res_agent['output']}")

except Exception as e:
    print(f"FATAL ERROR: {e}")
