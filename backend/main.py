import os
import time
import shutil
import pandas as pd
import numpy as np
print(f"Pandas Version: {pd.__version__}")
print(f"NumPy Version: {np.__version__}")
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv

# Non-interactive backend for plots
matplotlib.use('Agg')

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class AppState:
    vector_store = None
    dataframe = None
    agent = None
    # Load from environment variables
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

state = AppState()

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    # 1. Try Local Absolute Path (Dev)
    demo_path = r"d:/rate/Demo資料_利率合理性分析.xlsx"
    
    # 2. If not found, try Relative Path (Prod/Render)
    if not os.path.exists(demo_path):
        demo_path = "demo_data.xlsx"

    if os.path.exists(demo_path):
        print(f"Pre-loading demo file: {demo_path}")
        try:
            # Load specific sheet "Demo資料"
            df = pd.read_excel(demo_path, sheet_name="Demo資料")
            state.dataframe = df
            
            # Initialize Agent
            llm = AzureChatOpenAI(
                azure_deployment=state.chat_deployment,
                openai_api_version=state.api_version,
                azure_endpoint=state.endpoint,
                api_key=state.api_key,
                temperature=0
            )
            
            state.agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                verbose=True, 
                allow_dangerous_code=True,
                agent_executor_kwargs={"handle_parsing_errors": True},
                prefix=f"You are a detailed and professional AI Credit Analyst. The dataframe is loaded in `df`. \n\nCAPABILITIES:\n1. [METHOD EXPLANATION] If asked about 'How to perform reasonableness analysis?' (e.g., '請說明授信條件合理性分析'), DO NOT plot. Provide a structured text explanation with **Bold Headers** and *Bullet Points*. Sections: **利率**, **成數**, **寬限期**, **手續費**, **擔保品**, **還款方式**.\n2. [ANALYSIS & PLOTTING] If asked to 'Analyze rate' or for 'Reasonableness Analysis' on CURRENT data: 1. `import seaborn as sns; sns.set_theme(style='whitegrid')` 2. `import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']` 3. `plt.rcParams['axes.unicode_minus'] = False` 4. `plt.figure(figsize=(12, 8))` 5. Plot distribution. 6. `plt.savefig('temp/plot.png')` (MUST EXECUTE). In Final Answer: Provide a detailed analysis summary. DO NOT mention the file path 'temp/plot.png'.\n3. [OUTLIER CHECK] If asked about 'outliers' (離群值): You MUST use the `python_repl_ast` tool. Business Rule: Rates < 2.6% are considered outliers. Logic: 1. Filter rows where '利率' < 2.6. 2. In these rows, count occurrences of '新青安' (New Green Housing), '行員' (Bank Employee), and '利益關係人' (Stakeholder) by checking columns like '產品名稱', '備註', or '身分別'. 3. Report the counts for each category to explain why these rates are low.\n4. [DATA EXPORT] If asked to export details (e.g., 'Provide stakeholder details'): Filter rows where '利率' < 2.6 AND the row contains '利益關係人' (check '產品名稱', '備註', '身分別'). Save to 'temp/export.xlsx', and print the first 5 rows using `print(df.head().to_markdown(index=False, numalign='left', stralign='left'))`. Return detailed summary/confirmation."
            )
            print("Demo data loaded successfully!")
        except Exception as e:
            print(f"Failed to load demo data: {e}")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Backend is running. Please check /docs for API documentation."}

@app.get("/config-status")
async def get_config_status():
    return {
        "configured": all([state.api_key, state.endpoint, state.chat_deployment, state.embedding_deployment]),
        "endpoint": state.endpoint,
        "chat_model": state.chat_deployment,
        "demo_data_loaded": state.dataframe is not None
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not state.api_key:
        raise HTTPException(status_code=500, detail="Server not configured. Please check .env file.")
    
    try:
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Reset state
        state.vector_store = None
        state.dataframe = None
        state.agent = None

        if file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            # Excel -> Pandas Agent + RAG
            df = pd.read_excel(file_path)
            state.dataframe = df
            
            # Initialize Pandas Agent
            llm = AzureChatOpenAI(
                azure_deployment=state.chat_deployment,
                openai_api_version=state.api_version,
                azure_endpoint=state.endpoint,
                api_key=state.api_key,
                temperature=0
            )
            state.agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                verbose=True, 
                allow_dangerous_code=True,
                agent_executor_kwargs={"handle_parsing_errors": True},
                prefix=f"You are a detailed and professional AI Credit Analyst. The dataframe is loaded in `df`. \n\nCAPABILITIES:\n1. [METHOD EXPLANATION] If asked about 'How to perform reasonableness analysis?' (e.g., '請說明授信條件合理性分析'), DO NOT plot. Provide a structured text explanation with **Bold Headers** and *Bullet Points*. Sections: **利率**, **成數**, **寬限期**, **手續費**, **擔保品**, **還款方式**.\n2. [ANALYSIS & PLOTTING] If asked to 'Analyze rate' or for 'Reasonableness Analysis' on CURRENT data: 1. `import seaborn as sns; sns.set_theme(style='whitegrid')` 2. `import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']` 3. `plt.rcParams['axes.unicode_minus'] = False` 4. `plt.figure(figsize=(12, 8))` 5. Plot distribution. 6. `plt.savefig('temp/plot.png')` (MUST EXECUTE). In Final Answer: Provide a detailed analysis summary. DO NOT mention the file path 'temp/plot.png'.\n3. [OUTLIER CHECK] If asked about 'outliers' (離群值): You MUST use the `python_repl_ast` tool. Business Rule: Rates < 2.6% are considered outliers. Logic: 1. Filter rows where '利率' < 2.6. 2. In these rows, count occurrences of '新青安' (New Green Housing), '行員' (Bank Employee), and '利益關係人' (Stakeholder) by checking columns like '產品名稱', '備註', or '身分別'. 3. Report the counts for each category to explain why these rates are low.\n4. [DATA EXPORT] If asked to export details (e.g., 'Provide stakeholder details'): Filter rows where '利率' < 2.6 AND the row contains '利益關係人' (check '產品名稱', '備註', '身分別'). Save to 'temp/export.xlsx', and print the first 5 rows using `print(df.head().to_markdown(index=False, numalign='left', stralign='left'))`. Return detailed summary/confirmation."
            )

            # Also create RAG for semantic search (row by row)
            documents = []
            for index, row in df.iterrows():
                content_list = []
                for col in df.columns:
                    content_list.append(f"{col}: {row[col]}")
                content = "; ".join(content_list)
                documents.append(Document(page_content=content, metadata={"source": file.filename, "row": index}))
            
        elif file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        else:
            loader = TextLoader(file_path)
            documents = loader.load()
            
        # Common RAG Setup
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=state.embedding_deployment,
                openai_api_version=state.api_version,
                azure_endpoint=state.endpoint,
                api_key=state.api_key
            )
            state.vector_store = FAISS.from_documents(texts, embeddings)

        return {"message": "File processed successfully. You can now Ask about Policy (RAG) or Data (Analysis)."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    if not state.vector_store and not state.agent:
        raise HTTPException(status_code=400, detail="No document processed yet")
    
    try:
        response_data = {"response": ""}

        # Decision Logic: Simple heuristic
        # If we have an agent and query implies calculation/plot/export -> Agent
        # Else -> RAG
        
        use_agent = False
        if state.agent:
            keywords = ["analyze", "plot", "chart", "graph", "calculate", "count", "average", "sum", "export", "file", "excel", "outlier", "distribution", "合理性", "分析", "統計", "趨勢", "明細", "detail", "list", "output", "輸出"]
            if any(k in request.message.lower() for k in keywords):
                use_agent = True

        if use_agent:
            # Run Agent
            # Ensure temp dir exists
            os.makedirs("temp", exist_ok=True)
            
            # Clean up previous plot
            if os.path.exists("temp/plot.png"):
                try:
                    os.remove("temp/plot.png")
                except OSError:
                    pass
                
            res = state.agent.invoke(request.message)
            agent_output = res['output']
            
            # Check for plot
            if os.path.exists("temp/plot.png"):
                print("Found plot.png, attaching to response...")
                with open("temp/plot.png", "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                response_data["image"] = img_base64
                agent_output += "\n\n(Chart generated)"
            else:
                print("No plot.png found after agent execution.")
            
            # Check for export
            # Only trigger if export.xlsx exists and was modified recently (after request started)
            if os.path.exists("temp/export.xlsx"):
                 file_mtime = os.path.getmtime("temp/export.xlsx")
                 current_time = time.time()
                 # If modified within the last 30 seconds (generous buffer)
                 if current_time - file_mtime < 30:
                     response_data["file"] = "/api/download/export.xlsx"
                     agent_output += "\n\n(File ready for download)"

            response_data["response"] = agent_output

        else:
            # Run RAG with LCEL
            llm = AzureChatOpenAI(
                azure_deployment=state.chat_deployment,
                openai_api_version=state.api_version,
                azure_endpoint=state.endpoint,
                api_key=state.api_key,
                temperature=0
            )

            template = """You are a professional AI Credit Analyst (AI 智能授信分析師). 
Your goal is to answer questions based on the provided context or help with general inquiries.

Context:
{context}

Question: {question}

Instructions:
1. If the answer is in the Context, answer based on it.
2. If the user is just greeting (e.g., "Hi", "Hello", "你好") or engaging in casual chat, reply naturally, professionally, and politely in Traditional Chinese.
3. If the question is about data analysis but not found in context, you can say you don't have that specific info but suggest using the analysis tools.
4. Always maintain a professional yet helpful tone (Cool & Professional).
"""
            prompt = ChatPromptTemplate.from_template(template)
            
            if not state.vector_store:
                 # Fallback if no vector store (e.g. only Excel loaded)
                 # We can either return a polite message or try to use the Agent as fallback if available
                 if state.agent:
                     res = state.agent.invoke(request.message)
                     response_data["response"] = res['output']
                     return response_data
                 else:
                     raise HTTPException(status_code=400, detail="No knowledge base available for qualitative queries.")

            retriever = state.vector_store.as_retriever()

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            response_text = rag_chain.invoke(request.message)
            response_data["response"] = response_text

        return response_data
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("temp", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

# SERVE STATIC FILES (Frontend Build)
# We expect the frontend build to be in 'static' directory
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Use PORT env variable for Render
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
