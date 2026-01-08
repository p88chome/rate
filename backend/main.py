import os
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
    # Hardcoded path for Demo
    demo_path = r"d:/rate/Demo資料_利率合理性分析.xlsx"
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
                # Update prefix to ask for table preview and prettier plots + CHINESE FONT FIX + LARGER SIZE
                prefix="You are a data analyst. The dataframe is ALREADY loaded in the variable `df`. DO NOT try to read any excel file. Use `df` directly for analysis. If asked to plot, use `seaborn` with `sns.set_theme(style='whitegrid')` and a professional color palette. IMPORTANT: You MUST set `plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']` and `plt.rcParams['axes.unicode_minus'] = False` BEFORE plotting. Use `plt.figure(figsize=(12, 8))` to make the plot large and clear. save the plot to 'temp/plot.png'. If asked to export data, save to 'temp/export.xlsx'. If you export data, please also print the first 5 rows as a markdown table in your final answer. Return the filename in your final answer."
            )
            print("Demo data loaded successfully!")
        except Exception as e:
            print(f"Failed to load demo data: {e}")

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
                prefix="You are a data analyst. The dataframe is ALREADY loaded in the variable `df`. DO NOT try to read any excel file. Use `df` directly for analysis. If asked to plot, use `seaborn` with `sns.set_theme(style='whitegrid')` and a professional color palette. IMPORTANT: You MUST set `plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']` and `plt.rcParams['axes.unicode_minus'] = False` BEFORE plotting. Use `plt.figure(figsize=(12, 8))` to make the plot large and clear. save the plot to 'temp/plot.png'. If asked to export data, save to 'temp/export.xlsx'. If you export data, please also print the first 5 rows as a markdown table in your final answer. Return the filename in your final answer."
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
            # Clean up previous plot
            if os.path.exists("temp/plot.png"):
                os.remove("temp/plot.png")
                
            res = state.agent.invoke(request.message)
            agent_output = res['output']
            
            # Check for plot
            if os.path.exists("temp/plot.png"):
                with open("temp/plot.png", "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                response_data["image"] = img_base64
                agent_output += "\n\n(Chart generated)"
            
            # Check for export
            # We look if the agent said it saved a file or we check if a new .xlsx exists
            # For simplicity, let's assume if 'export.xlsx' is fresh? 
            # Or simplified: if user asked for export, check for 'temp/export.xlsx'
            # Check for export
            # We look if the agent said it saved a file or we check if a new .xlsx exists
            export_keywords = ["export", "excel", "輸出", "明細", "下載"]
            if any(k in request.message.lower() for k in export_keywords):
                 if os.path.exists("temp/export.xlsx"):
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

            template = """Answer the question based only on the following context:
{context}

Question: {question}
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
