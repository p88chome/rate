import os
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import requests
from matplotlib import font_manager

# Init Router
api_router = APIRouter()
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

# Force load .env from the same directory as main.py
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
loaded = load_dotenv(env_path)
print(f"Loading .env from: {env_path}")
print(f"Env loaded: {loaded}")
print(f"API Key present: {'Yes' if os.getenv('AZURE_OPENAI_API_KEY') else 'No'}")
print(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")

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
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "")
    font_setup_code: str = "import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']"

state = AppState()

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    # 1. Try Local Absolute Path (Dev)
    # Startup Config
    print(f"Startup Config: KeyLen={len(state.api_key) if state.api_key else 0}, Endpoint={state.endpoint}, Deploy={state.chat_deployment}")

    # 0. Check/Download Chinese Font (for Render/Linux)
    # 0. Check/Download Chinese Font (for Render/Linux)
    # Using jf-openhuninn-1.1.ttf (TrueType) for better compatibility on Linux/Render than OTF
    font_filename = "jf-openhuninn-1.1.ttf"
    if not os.path.exists(font_filename):
        print("Downloading Chinese font (jf-openhuninn)...")
        try:
            url = "https://github.com/justfont/open-huninn-font/blob/master/font/jf-openhuninn-1.1.ttf?raw=true"
            r = requests.get(url, allow_redirects=True)
            with open(font_filename, "wb") as f:
                f.write(r.content)
            print("Font downloaded.")
        except Exception as e:
            print(f"Failed to download font: {e}")
            font_filename = None
    
    # Store font path in state
    if font_filename and os.path.exists(font_filename):
        abs_font_path = os.path.abspath(font_filename).replace("\\", "/")
        state.font_setup_code = (
            f"import matplotlib.font_manager as fm; "
            f"fm.fontManager.addfont('{abs_font_path}'); "
            f"font_name = fm.FontProperties(fname='{abs_font_path}').get_name(); "
            f"plt.rcParams['font.sans-serif'] = [font_name, 'Microsoft JhengHei', 'sans-serif']; "
            f"plt.rcParams['axes.unicode_minus'] = False"
        )
    else:
        state.font_setup_code = "import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']; plt.rcParams['axes.unicode_minus'] = False"

    demo_path = r"d:/rate/Demo資料_利率合理性分析.xlsx"
    
    # 2. If not found, try Relative Path (Prod/Render)
    if not os.path.exists(demo_path):
        demo_path = "demo_data.xlsx"

    if os.path.exists(demo_path):
        print(f"Pre-loading demo file: {demo_path}")
        print(f"Startup Config: KeyLen={len(state.api_key) if state.api_key else 0}, Endpoint={state.endpoint}, Deploy={state.chat_deployment}")
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
                prefix=f"You are a detailed and professional AI Credit Analyst. The dataframe is ALREADY LOADED as `df`. DO NOT import pandas. DO NOT read files.\n\nCAPABILITIES:\n1. [METHOD EXPLANATION] If asked 'How to perform reasonableness analysis?' or '說明授信條件合理性' or '怎麼執行': \n   - DO NOT plot. Provide structured text: **利率**, **成數**, **寬限期**, **手續費**, **擔保品**, **還款方式**.\n2. [ANALYSIS & PLOTTING] If asked to 'Analyze rate' OR 'Scatter plot' OR 'Distribution' OR '母體分析': \n   - Step 1: CALCULATE stats first: `print(df['利率'].describe())`\n   - Step 2: EXECUTE PLOT (MANDATORY): \n     - IF 'scatter' or '散布' or '關係' or '關聯' in query: `import seaborn as sns; sns.set_theme(style='whitegrid'); import matplotlib.pyplot as plt; {state.font_setup_code}; plt.rcParams['axes.unicode_minus'] = False; plt.figure(figsize=(10,6)); sns.scatterplot(data=df, x='餘額', y='利率', alpha=0.6); plt.title('利率 vs 餘額 散布圖'); plt.savefig('temp/plot.png'); print('Plot saved to temp/plot.png')`\n     - ELSE (default distribution): `import seaborn as sns; sns.set_theme(style='whitegrid'); import matplotlib.pyplot as plt; {state.font_setup_code}; plt.rcParams['axes.unicode_minus'] = False; plt.figure(figsize=(10,6)); sns.histplot(df['利率'], bins=30, kde=True); plt.title('利率分布圖'); plt.savefig('temp/plot.png'); print('Plot saved to temp/plot.png')` \n   - Step 3: SUMMARIZE based on Step 1 stats. \n   - **IMPORTANT**: You MUST Run Step 2 to create the file. DO NOT output the raw Python code. DO NOT mention the file path 'temp/plot.png' or say 'Plot saved'. Just present the analysis.\n3. [OUTLIER CHECK] If asked about 'outliers' (離群值) OR '檢視利率是否合理' OR '合規' OR '銀行法': \n   - USE `python_repl_ast` IMMEDIATELY with this logic:\n   - `df_low = df[df['利率'] < 2.6]`\n   - `df_high = df[df['利率'] > 3.2]`\n   - `n_green = df_low['新青安註記'].notna().sum()` (Standardize check: Count non-nulls)\n   - `n_emp = df_low['行員註記'].notna().sum()`\n   - `n_stake = df_low['利益關係人'].notna().sum()`\n   - `print('Low:<2.6=', len(df_low), ' High:>3.2=', len(df_high), ' Green=', n_green, ' Emp=', n_emp, ' Stake=', n_stake)`\n   - **CRITICAL ANALYSIS**: Report Total Outliers (Low + High). For Low Rate: '新青安' & '行員' are reasonable. BUT for **Stakeholder (利益關係人)**, citing **Bank Act 33**, flag as **Potential Compliance Risk**. Mention High Rate count as well.\n4. [DATA EXPORT] If asked for 'Stakeholder details' or '利益關係人明細': \n   - Filter `df` where '利率' < 2.6 AND '利益關係人' IS NOT NULL (has value like 'V' or 'Y').\n   - Save to 'temp/export.xlsx'.\n   - Print first 5 rows with `print(df.head().to_markdown(index=False))`."
            )
            print("Demo data loaded successfully!")
        except Exception as e:
            print(f"Failed to load demo data: {e}")

@api_router.get("/")
async def root():
    return {"status": "ok", "message": "Backend is running. Please check /docs for API documentation."}

@api_router.get("/config-status")
async def get_config_status():
    return {
        "configured": all([state.api_key, state.endpoint, state.chat_deployment, state.embedding_deployment]),
        "endpoint": state.endpoint,
        "chat_model": state.chat_deployment,
        "demo_data_loaded": state.dataframe is not None
    }

@api_router.post("/upload")
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
                prefix=f"You are a detailed and professional AI Credit Analyst. The dataframe is ALREADY LOADED as `df`. DO NOT import pandas. DO NOT read files.\n\nCAPABILITIES:\n1. [METHOD EXPLANATION] If asked 'How to perform reasonableness analysis?' or '說明授信條件合理性' or '怎麼執行': \n   - DO NOT plot. Provide structured text: **利率**, **成數**, **寬限期**, **手續費**, **擔保品**, **還款方式**.\n2. [ANALYSIS & PLOTTING] If asked to 'Analyze rate' OR 'Scatter plot' OR 'Distribution' OR '母體分析': \n   - Step 1: CALCULATE stats first: `print(df['利率'].describe())`\n   - Step 2: EXECUTE PLOT (MANDATORY): \n     - IF 'scatter' or '散布' or '關係' or '關聯' in query: `import seaborn as sns; sns.set_theme(style='whitegrid'); import matplotlib.pyplot as plt; {state.font_setup_code}; plt.rcParams['axes.unicode_minus'] = False; plt.figure(figsize=(10,6)); sns.scatterplot(data=df, x='餘額', y='利率', alpha=0.6); plt.title('利率 vs 餘額 散布圖'); plt.savefig('temp/plot.png'); print('Plot saved to temp/plot.png')`\n     - ELSE (default distribution): `import seaborn as sns; sns.set_theme(style='whitegrid'); import matplotlib.pyplot as plt; {state.font_setup_code}; plt.rcParams['axes.unicode_minus'] = False; plt.figure(figsize=(10,6)); sns.histplot(df['利率'], bins=30, kde=True); plt.title('利率分布圖'); plt.savefig('temp/plot.png'); print('Plot saved to temp/plot.png')` \n   - Step 3: SUMMARIZE based on Step 1 stats. \n   - **IMPORTANT**: You MUST Run Step 2 to create the file. DO NOT output the raw Python code. DO NOT mention the file path 'temp/plot.png' or say 'Plot saved'. Just present the analysis.\n3. [OUTLIER CHECK] If asked about 'outliers' (離群值) OR '檢視利率是否合理' OR '合規' OR '銀行法': \n   - USE `python_repl_ast` IMMEDIATELY with this logic:\n   - `df_low = df[df['利率'] < 2.6]`\n   - `df_high = df[df['利率'] > 3.2]`\n   - `n_green = df_low['新青安註記'].notna().sum()` (Standardize check: Count non-nulls)\n   - `n_emp = df_low['行員註記'].notna().sum()`\n   - `n_stake = df_low['利益關係人'].notna().sum()`\n   - `print('Low:<2.6=', len(df_low), ' High:>3.2=', len(df_high), ' Green=', n_green, ' Emp=', n_emp, ' Stake=', n_stake)`\n   - **CRITICAL ANALYSIS**: Report Total Outliers (Low + High). For Low Rate: '新青安' & '行員' are reasonable. BUT for **Stakeholder (利益關係人)**, citing **Bank Act 33**, flag as **Potential Compliance Risk**. Mention High Rate count as well.\n4. [DATA EXPORT] If asked for 'Stakeholder details' or '利益關係人明細': \n   - Filter `df` where '利率' < 2.6 AND '利益關係人' IS NOT NULL (has value like 'V' or 'Y').\n   - Save to 'temp/export.xlsx'.\n   - Print first 5 rows with `print(df.head().to_markdown(index=False))`."
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
            
            print(f"Initializing Embeddings with: Deployment={state.embedding_deployment}, Endpoint={state.endpoint}, Key={'Yes' if state.api_key else 'No'}")
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=state.embedding_deployment,
                openai_api_version=state.api_version,
                azure_endpoint=state.endpoint,
                api_key=state.api_key
            )
            state.vector_store = FAISS.from_documents(texts, embeddings)
            print("Startup Complete: Agent and Vector Store initialized.")

        return {"message": "File processed successfully. You can now Ask about Policy (RAG) or Data (Analysis)."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/chat")
async def chat(request: ChatRequest):
    if not state.vector_store and not state.agent:
        raise HTTPException(status_code=400, detail="No document processed yet")
    
    print(f"Chat Request: Agent={bool(state.agent)}, RAG={bool(state.vector_store)}")
    print(f"Runtime Config: KeyLen={len(state.api_key) if state.api_key else 0}, Model={state.chat_deployment}")

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
        
        print(f"Decision: Use Agent? {use_agent}")

        if use_agent:
            print("Invoking Agent...")
            # Ensure temp dir exists
            os.makedirs("temp", exist_ok=True)
            
            # Clean up previous plot
            if os.path.exists("temp/plot.png"):
                try:
                    os.remove("temp/plot.png")
                except OSError:
                    pass

            try:
                res = state.agent.invoke(request.message)
                print("Agent invoke success")
                agent_output = res['output']
                
                # Check for plot
                if os.path.exists("temp/plot.png"):
                    print("Found plot.png, attaching to response...")
                    with open("temp/plot.png", "rb") as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                    response_data["image"] = img_base64
                    # Removed redundant text: agent_output += "\n\n(Chart generated)"
                else:
                    # Fallback: If intent was to plot (keywords check) but file not found -> Force Plot
                    plot_keywords = ["plot", "chart", "graph", "distribution", "圖", "分布", "畫", "draw", "scatter", "散布", "母體"]
                    is_scatter = any(k in request.message.lower() for k in ["scatter", "散布", "關係", "關聯"])
                    
                    if any(k in request.message.lower() for k in plot_keywords) and state.dataframe is not None:
                         print(f"DEBUG: Plot intent detected (scatter={is_scatter}) but no file from Agent. Attempting Force Plot...")
                         plot_type = 'scatter' if is_scatter else 'hist'
                         if force_generate_plot(state.dataframe, plot_type):
                              with open("temp/plot.png", "rb") as f:
                                  img_base64 = base64.b64encode(f.read()).decode('utf-8')
                              response_data["image"] = img_base64
                              # Removed redundant text: agent_output += "\n\n(Chart generated by System Fallback)"
                         else:
                              print("DEBUG: Force Plot failed.")

                # Check for export
                if os.path.exists("temp/export.xlsx"):
                     print("DEBUG: found temp/export.xlsx immediately after agent run")
                     file_mtime = os.path.getmtime("temp/export.xlsx")
                     current_time = time.time()
                     # If modified within the last 30 seconds (generous buffer)
                     if current_time - file_mtime < 30:
                         print("DEBUG: file is fresh, attaching download link")
                         response_data["file"] = "/api/download/export.xlsx"
                         # agent_output += "\n\n(File ready for download)"
                     else:
                         print(f"DEBUG: file is stale (age={current_time - file_mtime}s)")
                else:
                     print("DEBUG: temp/export.xlsx NOT FOUND after agent run")

                response_data["response"] = agent_output
            except Exception as inner_e:
                print(f"AGENT ERROR: {inner_e}")
                import traceback
                traceback.print_exc()
                # Return the error to the user so they (and we) can see it in valid JSON
                response_data["response"] = f"Agent Execution Error: {str(inner_e)}"

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

def force_generate_plot(df, plot_type='hist'):
    """Fallback function to generate a plot if Agent fails."""
    try:
        print(f"FORCE PLOT: Generating fallback plot type={plot_type}...")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.close('all') # Clear any existing plots
        
        # Execute Setup Code for Fonts (Safe Mode)
        try:
            exec(state.font_setup_code, globals())
        except:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
        
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10,6))
        sns.set_theme(style='whitegrid')
        
        if plot_type == 'scatter':
            # Scatter Plot: Rate vs LTV or Amount
            # Candidates for X-axis
            x_candidates = ['成數', '核貸成數', '貸款成數', 'LTV', '核准金額', '授信金額', '貸款金額', '金額', '餘額']
            x_col = next((col for col in x_candidates if col in df.columns), None)
            
            if '利率' in df.columns and x_col:
                # Re-apply font settings to be safe
                try:
                    exec(state.font_setup_code, globals())
                except:
                    pass
                
                sns.scatterplot(data=df, x=x_col, y='利率', alpha=0.6)
                plt.title(f'利率 vs {x_col} 散布圖')
                plt.savefig('temp/plot.png')
                print(f"FORCE PLOT: Saved temp/plot.png (Scatter: {x_col})")
                return True
            else:
                print(f"FORCE PLOT: No suitable X column found for scatter. Candidates={x_candidates}. Falling back to hist.")
                # Fall through to hist
        
        # Determine what to plot based on columns
        if '利率' in df.columns:
             # Re-apply font settings to be safe
             try:
                 exec(state.font_setup_code, globals())
             except:
                 pass
             
             sns.histplot(df['利率'], bins=30, kde=True)
             plt.title('利率分布圖')
             plt.savefig('temp/plot.png')
             print("FORCE PLOT: Saved temp/plot.png (Hist)")
             return True
        else:
             print("FORCE PLOT: '利率' column not found.")
             return False
    except Exception as e:
        print(f"FORCE PLOT ERROR: {e}")
        return False

@api_router.get("/download/{filename}")
async def download_file(filename: str):
    print(f"DOWNLOAD REQUEST: filename={filename}")
    # Security check
    if ".." in filename or "/" in filename or "\\" in filename:
        print(f"DOWNLOAD ERROR: Invalid filename {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    file_path = os.path.join("temp", filename)
    exists = os.path.exists(file_path)
    print(f"DOWNLOAD CHECK: path={file_path}, exists={exists}")
    
    if exists:
        return FileResponse(file_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")

# EXPLICIT FALLBACK ROUTE for Render compatibility
@app.get("/api/download/{filename}")
async def download_file_api(filename: str):
    print(f"EXPLICIT API DOWNLOAD REQUEST: {filename}")
    return await download_file(filename)

# Include Router Twice for Compatibility
app.include_router(api_router)
app.include_router(api_router, prefix="/api")

# SERVE STATIC FILES (Frontend Build)
# We expect the frontend build to be in 'static' directory
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Use PORT env variable for Render
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
