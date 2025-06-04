import streamlit as st
import os
import time
import uuid
from dotenv import load_dotenv

# LangChain, Google, Pinecone imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone as PineconeClient # Pinecone's own client
from langchain_pinecone import Pinecone as LangchainPineconeStore # LangChain's wrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# W&B import if you want to log UI interactions as well
# import wandb # Keep this if you might uncomment W&B parts

st.set_page_config(page_title="Document Q&A Assistant", layout="wide")

# --- 0. Load Environment Variables & Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Your Pinecone region
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Optional W&B configuration for UI logs
# WANDB_API_KEY = os.getenv("WANDB_API_KEY")
# WANDB_PROJECT_NAME_UI = os.getenv("WANDB_PROJECT_NAME_UI", "gemini-doc-assistant-ui")

# Model and Embedding Configuration
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "models/embedding-001"
# EMBEDDING_DIMENSION = 768 # Pinecone index should match this

# --- App Configuration ---
TEMP_UPLOAD_DIR = "temp_ui_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- Initialize W&B (Optional for UI) ---
# def initialize_wandb_ui():
#     if WANDB_API_KEY:
#         wandb.login(key=WANDB_API_KEY)
#     wandb.init(
#         project=WANDB_PROJECT_NAME_UI,
#         name=f"ui-session-{time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:4]}",
#         config={"llm_model": LLM_MODEL_NAME, "embedding_model": EMBEDDING_MODEL_NAME},
#         job_type="interactive_ui"
#     )
#     print(f"W&B initialized for UI project: {WANDB_PROJECT_NAME_UI}")

# if WANDB_API_KEY: # Initialize only if key is present
#    initialize_wandb_ui()


# --- Core Component Initialization (Cached) ---
@st.cache_resource # Caches the resource across reruns for the same session
def initialize_core_components():
    """Initializes and returns core LangChain and Pinecone components."""
    st.write("Attempting to initialize core components...") # For debugging in UI
    print("Attempting to initialize core components for Streamlit UI...")
    if not all([GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
        st.error("FATAL: Critical API keys or Pinecone configuration is missing from environment variables. Cannot initialize components.")
        print("FATAL: Critical API keys or Pinecone configuration is missing. Cannot initialize components.")
        return None, None, None, None # Return None for all if critical components fail

    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY
        )
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=0.3, # Slightly higher for more conversational UI answers
            google_api_key=GOOGLE_API_KEY
        )

        # Initialize Pinecone client (Pinecone's own client, not LangChain's yet)
        pinecone_native_client = PineconeClient(api_key=PINECONE_API_KEY)
        
        # Check if index exists, if not, inform user (UI cannot create index easily)
        # Get list of index names by iterating through the IndexList object
        active_indexes = [index_info.name for index_info in pinecone_native_client.list_indexes()]
        if PINECONE_INDEX_NAME not in active_indexes:
            st.error(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Please ensure the index is created and configured correctly (e.g., via main_api.py or Pinecone console).")
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' not found.")
            return None, None, None, None

        pinecone_index_object = pinecone_native_client.Index(PINECONE_INDEX_NAME)

        # Initialize LangChain's Pinecone vector store
        # This object will be used for retrieval and adding new documents
        langchain_pinecone_store = LangchainPineconeStore(
            index=pinecone_index_object, # Pass the Pinecone index object from the native client
            embedding=embeddings_model,
            text_key="text" # Assumes text is stored in metadata field 'text'
        )

        retriever = langchain_pinecone_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks
        
        prompt_template_str = """You are a helpful AI assistant answering questions based on the provided context.
        Context:
        {context}

        Question: {question}

        Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        st.success("Core components initialized successfully!")
        print("Core components initialized successfully for Streamlit UI.")
        return qa_chain, embeddings_model, pinecone_index_object, langchain_pinecone_store

    except Exception as e:
        st.error(f"Error during core component initialization: {e}")
        print(f"Error during core component initialization: {e}")
        return None, None, None, None


# Attempt to initialize components when the script first runs
# These will be cached by Streamlit
qa_chain, embeddings, pinecone_index, lc_pinecone_store = initialize_core_components()


# --- PDF Processing and Embedding Function ---
def process_and_embed_pdf(uploaded_file_obj, p_index, embed_model, lc_store):
    """Processes an uploaded PDF, embeds its content, and adds to Pinecone."""
    if uploaded_file_obj is None:
        return False

    file_name = uploaded_file_obj.name
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{str(uuid.uuid4())}_{file_name}")

    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file_obj.getbuffer())
        
        st.info(f"Processing '{file_name}'...")
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        if not documents:
            st.warning(f"Could not extract text from '{file_name}'.")
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents) # List of LangChain Document objects

        if not texts:
            st.warning(f"No text chunks to process from '{file_name}'.")
            return False

        st.info(f"Embedding {len(texts)} chunks from '{file_name}' for Pinecone...")
        
        lc_store.add_documents(texts) # This uses the LangChain Pinecone store
        
        st.success(f"Successfully processed and added '{file_name}' to the knowledge base.")
        return True

    except Exception as e:
        st.error(f"Error processing PDF '{file_name}': {e}")
        print(f"Error processing PDF '{file_name}': {e}")
        return False
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path) # Clean up temporary file

# --- Streamlit App Layout ---
st.header("📄 Document Q&A Assistant")

if not qa_chain or not lc_pinecone_store: # Check if components initialized
    st.error("Application is not initialized correctly. Please check environment variables and ensure Pinecone index exists. Then, refresh the page.")
    st.stop() # Stop execution if core components failed

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.subheader("Upload Documents")
    st.markdown("Add PDF files to the knowledge base for Q&A.")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True, key="pdf_uploader"
    )

    if uploaded_files:
        # Use a button to trigger processing to avoid re-processing on every interaction
        if st.button("Process Uploaded Files", key="process_button"):
            with st.spinner("Processing files... This may take a moment."):
                num_processed = 0
                for up_file in uploaded_files:
                    # Simple check to avoid reprocessing if file name was already processed
                    if f"processed_{up_file.name}" not in st.session_state:
                        if process_and_embed_pdf(up_file, pinecone_index, embeddings, lc_pinecone_store):
                            st.session_state[f"processed_{up_file.name}"] = True
                            num_processed += 1
                    else:
                        st.info(f"'{up_file.name}' seems to have been processed already in this session.")
                if num_processed > 0:
                    st.success(f"Processed {num_processed} new file(s).")
                else:
                    st.info("No new files were processed, or files were already processed this session.")


# --- Main Q&A Interface ---
st.subheader("Ask Questions About Your Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload some PDFs using the sidebar and ask me anything about them."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Your question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                answer = result.get("result", "Sorry, I could not find an answer.")
                
                sources_text = "" # Initialize as empty
                unique_filenames = set()

                if "source_documents" in result and result["source_documents"]:
                    for doc_source in result["source_documents"]:
                        # Try to get 'original_filename' from metadata, which we aim to set during upload processing.
                        # Fallback to basename of 'source' if 'original_filename' isn't there.
                        source_path = doc_source.metadata.get("source", "Unknown source")
                        filename = doc_source.metadata.get("original_filename", os.path.basename(source_path))
                        if filename != "Unknown source": # Avoid adding "Unknown source" if path is truly unknown
                            unique_filenames.add(filename)
                
                if unique_filenames:
                    sources_text = "\n\n**Sources:**\n"
                    for name in sorted(list(unique_filenames)): # Sort for consistent order
                        sources_text += f"- {name}\n"
                else:
                    sources_text = "\n\n(No specific source documents identified for this answer.)"

                full_response_content = answer + sources_text
                message_placeholder.markdown(full_response_content)
                
            except Exception as e:
                full_response_content = f"Sorry, an error occurred: {e}"
                message_placeholder.error(full_response_content)
                
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response_content})

st.markdown("---")
st.caption("Powered by LangChain, Google Gemini, Pinecone, and Streamlit.")
