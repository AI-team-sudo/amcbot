import streamlit as st
import openai
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore  # Updated import
from llama_index.llms.openai import OpenAI
import pinecone
import re

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Rag Based Bot for GPMC using Pinecone",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize API keys from Streamlit secrets
openai.api_key = st.secrets.get("openai_key", None)
PINECONE_API_KEY = st.secrets.get("pinecone_api_key", None)
PINECONE_ENVIRONMENT = st.secrets.get("pinecone_environment", "gcp-starter")

if not openai.api_key or not PINECONE_API_KEY:
    st.error("API keys are missing. Please add them to the Streamlit secrets.")
    st.stop()

st.title("Rag Based Bot for The Ahmedabad Municipal Corporation")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome! I can help you understand the GPMC Act and AMC procedures in detail. Please ask your question!"
        }
    ]

@st.cache_resource(show_spinner=False)
def initialize_pinecone():
    try:
        # Initialize Pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT
        )

        # Connect to the existing index
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone.Index("amcbots")
        )

        system_prompt = """You are an authoritative expert on the GPMC Act and the Ahmedabad Municipal Corporation.
        Your responses should be:
        1. Comprehensive and detailed
        2. Include step-by-step procedures when applicable
        3. Quote relevant sections directly from the GPMC Act
        4. Provide specific references (section numbers, chapters, and page numbers)
        5. Break down complex processes into numbered steps
        6. Include any relevant timelines or deadlines
        7. Mention any prerequisites or requirements
        8. Highlight important caveats or exceptions

        For every fact or statement, include a reference to the source document and section number."""

        Settings.llm = OpenAI(
            model="gpt-4",
            temperature=0.1,
            system_prompt=system_prompt,
        )

        index = VectorStoreIndex.from_vector_store(vector_store)
        return index
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        st.stop()

# Rest of the code remains the same...
