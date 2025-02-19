import streamlit as st
import openai
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
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

        # Create service context
        llm = OpenAI(
            model="gpt-4",
            temperature=0.1,
            system_prompt="""You are an authoritative expert on the GPMC Act and the Ahmedabad Municipal Corporation.
            Your responses should be comprehensive, detailed, and include relevant references."""
        )
        service_context = ServiceContext.from_defaults(llm=llm)

        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            service_context=service_context
        )
        return index
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        st.stop()

def format_response(response):
    formatted_response = response.replace("Step ", "\n### Step ")
    formatted_response = formatted_response.replace("Note:", "\n> **Note:**")
    formatted_response = formatted_response.replace("Important:", "\n> **Important:**")
    return formatted_response

# Initialize Pinecone and create chat engine
index = initialize_pinecone()

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True
    )

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This chatbot uses Pinecone vector database to provide accurate information about:
    - GPMC Act
    - AMC procedures
    - Municipal regulations
    - Administrative processes
    """)

# Chat interface
if prompt := st.chat_input("Ask a question about GPMC Act or AMC procedures"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Generate new response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        try:
            response = st.session_state.chat_engine.chat(prompt)
            formatted_response = format_response(response.response)

            st.markdown(formatted_response, unsafe_allow_html=True)

            message = {
                "role": "assistant",
                "content": formatted_response
            }
            st.session_state.messages.append(message)
        except Exception as e:
            st.error(f"Error generating response: {e}")

# Add CSS for better formatting
st.markdown("""
<style>
.stChat message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)
