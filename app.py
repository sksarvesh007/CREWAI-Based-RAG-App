import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Pinecone
index_name = "simple-rag-app"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(index_name)

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize BM25 encoder with persistent caching
@st.cache_data  # Cache persists in Streamlit's default location
def initialize_bm25_encoder():
    encoder = BM25Encoder().default()
    # Assuming 'bm25_vals.json' exists; update the path if needed
    encoder.load("bm25_vals.json")
    return encoder

bm25_encoder = initialize_bm25_encoder()

# Initialize PineconeHybridSearchRetriever
retriever = PineconeHybridSearchRetriever(
    index=index,
    embeddings=embeddings,
    sparse_encoder=bm25_encoder
)

def retrieve(query):
    """
    Retrieve relevant documents from the Pinecone index based on the query.
    """
    results = retriever.invoke(query)
    return results

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Streamlit App
st.title("RAG-based Chatbot with Memory")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input query
query = st.chat_input("Enter your query:")

if query:
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve relevant documents
    context = retrieve(query)

    # Create system prompt with chat history
    chat_history_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history]
    )
    system_prompt = f"""
    You are a helpful assistant. Use the following chat history and context to answer the user's question.
    
    Chat History:
    {chat_history_str}
    
    Context:
    {context}
    
    Question:
    {query}
    
    If the context mentions a section number, ensure your answer is relevant to that section and mention the section number.
    """

    # Stream the response
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=1,
            max_tokens=2048,
            top_p=1,
            stream=True,
            stop=None,
        )

        for chunk in completion:
            chunk_content = chunk.choices[0].delta.content or ""
            full_response += chunk_content
            response_container.markdown(full_response)

    # Add assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
