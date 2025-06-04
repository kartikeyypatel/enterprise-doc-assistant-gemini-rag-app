import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body, Request as FastAPIRequest # Import Request for middleware
from pydantic import BaseModel
import uuid # To generate unique IDs for requests

# LangChain and Google imports (as before)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from pinecone import Pinecone as PineconeClient, ServerlessSpec, PodSpec
from langchain_pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- ADD W&B Import ---
import wandb

# --- 0. Load Environment Variables & Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
print(f"DEBUG - Loaded PINECONE_INDEX_NAME: '{PINECONE_INDEX_NAME}'")
# --- ADD W&B API Key Env Var (Optional here if logged in, but good for explicitness) ---
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME", "gemini-doc-assistant") # Default project name

# Basic check for critical env vars
if not all([GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
    raise ValueError("Missing one or more critical environment variables for Google/Pinecone.")

DOCS_DIRECTORY = "docs"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "models/embedding-001"
EMBEDDING_DIMENSION = 768

# Global Variables (as before)
embeddings_model = None
llm = None
qa_chain = None
pinecone_index = None
vector_store = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Enterprise Document Intelligence Assistant",
    description="API for querying documents using a RAG pipeline with Gemini and Pinecone, monitored by W&B.",
    version="0.3.0" # Version bump
)

# --- W&B Initialization Function ---
def initialize_wandb():
    if WANDB_API_KEY: # Ensure API key is available if provided this way
        wandb.login(key=WANDB_API_KEY)

    # You can customize project name, entity, run name, etc.
    wandb.init(
        project=WANDB_PROJECT_NAME,
        # entity="your_wandb_username_or_team", # Optional: your W&B username or team name
        name=f"api-run-{time.strftime('%Y%m%d-%H%M%S')}", # Give each API run a unique name
        config={ # Log static configuration
            "llm_model": LLM_MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "pinecone_index": PINECONE_INDEX_NAME,
        },
        #reinit=True, # Allows re-initializing in the same process (useful for some server setups)
        job_type="api_service" # Categorize the run
    )
    print(f"Weights & Biases initialized for project: {WANDB_PROJECT_NAME}")

# --- Helper Functions (load_and_split_documents - no change) ---
def load_and_split_documents(directory_path): # (Same as before)
    print(f"Loading documents from: {directory_path}")
    loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
    documents = loader.load()
    if not documents:
        print(f"No PDF documents found in '{directory_path}'.")
        return []
    print(f"Loaded {len(documents)} document(s).")
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")
    return texts

# --- Modified initialize_pinecone_and_components (no direct W&B calls here, but W&B is init on app startup) ---
def initialize_pinecone_and_components(): # (Same as before, ensure it runs after W&B init if needed for config)
    global embeddings_model, llm, qa_chain, pinecone_index, vector_store
    print("Initializing embeddings model and LLM...")
    # ... (rest of the initialization logic for Google and Pinecone as before) ...
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY
        )
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )
    except Exception as e:
        raise RuntimeError(f"Could not initialize Google GenAI models: {e}")

    print(f"Initializing Pinecone client for environment: {PINECONE_ENVIRONMENT}...")
    pinecone_client_instance = PineconeClient(api_key=PINECONE_API_KEY)
    
    list_of_indexes_models = pinecone_client_instance.list_indexes()
    active_index_names = [index_model.name for index_model in list_of_indexes_models]

    if PINECONE_INDEX_NAME not in active_index_names:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating new index...")
        spec = ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT) # Adjust cloud and region
        try:
            pinecone_client_instance.create_index(
                name=PINECONE_INDEX_NAME, dimension=EMBEDDING_DIMENSION, metric="cosine", spec=spec
            )
            print(f"Waiting for index '{PINECONE_INDEX_NAME}' to be ready...")
            while not pinecone_client_instance.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(5)
            print(f"Index '{PINECONE_INDEX_NAME}' created and ready.")
            populate_pinecone_index = True
        except Exception as e:
            raise RuntimeError(f"Failed to create Pinecone index '{PINECONE_INDEX_NAME}': {e}")
    else:
        print(f"Found existing Pinecone index: '{PINECONE_INDEX_NAME}'")
        populate_pinecone_index = False

    pinecone_index = pinecone_client_instance.Index(PINECONE_INDEX_NAME)
    stats = pinecone_index.describe_index_stats()
    print(f"Pinecone index stats: {stats}")

    if populate_pinecone_index or (stats.total_vector_count == 0 and os.path.exists(DOCS_DIRECTORY) and os.listdir(DOCS_DIRECTORY)):
        print("Populating Pinecone index...")
        texts_to_embed = load_and_split_documents(DOCS_DIRECTORY)
        if texts_to_embed:
            print(f"Attempting to populate Pinecone using LangChain's from_documents for {len(texts_to_embed)} chunks...")
            try:
                vector_store = Pinecone.from_documents(
                    documents=texts_to_embed, embedding=embeddings_model, index_name=PINECONE_INDEX_NAME
                )
                print("Pinecone index populated using from_documents.")
            except Exception as e:
                 print(f"Error populating Pinecone with from_documents: {e}.")
                 vector_store = Pinecone(index=pinecone_index, embedding=embeddings_model, text_key="text")
        else:
            print("No documents found to populate Pinecone index.")
            vector_store = Pinecone(index=pinecone_index, embedding=embeddings_model, text_key="text")
    else:
        print("Initializing LangChain Pinecone vector store with existing index...")
        vector_store = Pinecone(index=pinecone_index, embedding=embeddings_model, text_key="text")

    if not vector_store:
        raise RuntimeError("Failed to initialize Pinecone vector store.")

    print("Setting up RAG QA chain with Pinecone retriever...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt_template_str = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer concise. Context: {context} Question: {question} Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("--- Document Intelligence Assistant API with Pinecone Ready (W&B Monitoring Enabled) ---")


# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    print("Application startup...")
    initialize_wandb() # Initialize W&B first
    initialize_pinecone_and_components() # Then other components
    print("Application initialization complete.")

# --- Pydantic Models (No change) ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: list = []

# --- API Endpoints ---
@app.post("/query/", response_model=QueryResponse)
async def ask_question(fastapi_req: FastAPIRequest, request: QueryRequest = Body(...)): # Added FastAPIRequest
    if not qa_chain:
        raise HTTPException(status_code=503, detail="QA chain is not initialized.")
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    request_id = str(uuid.uuid4()) # Generate a unique ID for this request
    start_time = time.time()
    print(f"Request ID {request_id}: Received query: {request.query}")

    try:
        # Log query to W&B
        wandb.log({
            "request_id": request_id,
            "query": request.query,
            "timestamp": time.time()
        })

        result = qa_chain.invoke({"query": request.query})
        answer = result["result"]
        
        sources_data = []
        if "source_documents" in result and result["source_documents"]:
            for doc in result["source_documents"]:
                sources_data.append({
                    "page_content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                })
        
        processing_time = time.time() - start_time

        # Log response and performance to W&B
        wandb.log({
            "request_id": request_id, # To link logs for the same request
            "answer": answer,
            "num_source_documents": len(sources_data),
            # "source_previews": [s["page_content_preview"] for s in sources_data], # Can be verbose
            "processing_time_seconds": processing_time,
            "query_successful": True
        })
        
        return QueryResponse(answer=answer, source_documents=sources_data)
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"Request ID {request_id}: Error during question answering: {e}")
        # Log error to W&B
        wandb.log({
            "request_id": request_id,
            "query": request.query,
            "error_message": str(e),
            "processing_time_seconds": processing_time,
            "query_successful": False
        })
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Enterprise Document Intelligence Assistant API (Pinecone Edition with W&B Monitoring)!"}

# To run locally (save as main_api.py):
# Ensure WANDB_API_KEY and WANDB_PROJECT_NAME are in your .env or environment
# uvicorn main_api:app --reload --port 8000