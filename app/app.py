# --- Streamlit Cloud sqlite fix for Chroma ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# --------------------------------------------



import streamlit as st
import os
import tempfile
import pandas as pd
import anthropic
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    st.error("Missing ANTHROPIC_API_KEY. Please set it in your environment or Streamlit Secrets.")
    st.stop()



# Customize initial app landing page
st.set_page_config(page_title="File QA Chatbot with Claude", page_icon="🤖")
st.title("Welcome to File QA RAG Chatbot with Claude 🤖")

# Create a sidebar for model selection
with st.sidebar:
    st.header("Configuration")

    # Model selection
    model_option = st.selectbox(
        "Select Claude Model",
        ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
        index=0
    )

    st.divider()

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split into document chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(docs)

    # Create document embeddings and store in Vector DB
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(doc_chunks, embeddings_model)

    # Define retriever object
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever

# Streaming handler for Claude responses
class StreamHandler:
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Function to get Claude's response
def get_claude_response(api_key, model, prompt, stream_handler=None):
    client = anthropic.Anthropic(api_key=api_key)

    if stream_handler:
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            temperature=0.2,
            system="You are a helpful AI assistant that answers questions based only on the provided context.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        ) as stream:
            for text in stream.text_stream:
                stream_handler.on_llm_new_token(text)

            return stream.get_final_message().content
    else:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.2,
            system="You are a helpful AI assistant that answers questions based only on the provided context.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content

# Creates UI element to accept PDF uploads
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"],
    accept_multiple_files=True
)

# Check if PDFs are uploaded
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

# Create retriever object based on uploaded PDFs
retriever = configure_retriever(uploaded_files)

# This function formats retrieved documents before sending to LLM
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Store conversation history in Streamlit session state
streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")

# Shows the first message when app starts
if len(streamlit_msg_history.messages) == 0:
    streamlit_msg_history.add_ai_message("Please ask your question about the uploaded documents?")

# Render current messages from StreamlitChatMessageHistory
for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, display it and show the response
if user_prompt := st.chat_input():
    # Add the user message to history
    streamlit_msg_history.add_user_message(user_prompt)

    # Display the user message
    st.chat_message("human").write(user_prompt)

    
    # LangChain retrievers: prefer invoke()
    docs = retriever.invoke(user_prompt)  # preferred



    # Store source documents for display
    sources = []
    source_ids = []
    for d in docs:
        metadata = {
            "source": d.metadata["source"],
            "page": d.metadata["page"],
            "content": d.page_content[:200]
        }
        idx = (metadata["source"], metadata["page"])
        if idx not in source_ids:
            source_ids.append(idx)
            sources.append(metadata)

    # Format the documents
    context = format_docs(docs)

    # Create the prompt
    qa_template = """
    Use only the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know,
    don't try to make up an answer. Keep the answer as concise as possible.

    Context:
    {context}

    Question: {question}
    """

    prompt = qa_template.format(context=context, question=user_prompt)

    # This is where response from Claude is shown
    with st.chat_message("ai"):
        # Initializing an empty data stream
        stream_container = st.empty()
        stream_handler = StreamHandler(stream_container)

        # Get Claude response with streaming
        response = get_claude_response(
            api_key=anthropic_api_key,
            model=model_option,
            prompt=prompt,
            stream_handler=stream_handler
        )

        # Add the AI response to history
        streamlit_msg_history.add_ai_message(stream_handler.text)

        # Display sources
        if len(sources):
            st.markdown("__Sources:__ "+"\n")
            st.dataframe(data=pd.DataFrame(sources[:3]), width=1000)  # Top 3 source
