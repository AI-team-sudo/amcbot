import streamlit as st
from pinecone import Pinecone
import openai
from typing import List

# Initialize OpenAI with st.secrets
openai.api_key = st.secrets["openai_key"]

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
index = pc.Index("amcbots")

def get_embedding(text: str) -> List[float]:
    """Get embedding for the input text using OpenAI's embedding model."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def search_pinecone(query: str, k: int = 3):
    """Search Pinecone index with embedded query."""
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    return results

def generate_response(query: str, context: str, system_prompt: str):
    """Generate response using OpenAI with context and system prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message['content']

# Streamlit UI
st.title("Gujarat Municipal Act Assistant")
st.write("Ask any question about the GPMC Act and Ahmedabad Municipal Corporation")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Search Pinecone and get relevant context
    search_results = search_pinecone(prompt)
    context = "\n".join([result.metadata.get('text', '') for result in search_results.matches])

    # Generate response
    with st.chat_message("assistant"):
        response = generate_response(prompt, context, system_prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.write("This chatbot provides information about the Gujarat Municipal Act and Ahmedabad Municipal Corporation.")
    st.write("It uses AI to search through the official documentation and provide accurate responses with references.")
