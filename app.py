import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from typing import List
from deep_translator import GoogleTranslator
import time
import math

# Debug mode
DEBUG = st.sidebar.checkbox("Debug Mode", False)

# System prompt - modified for AMC GPMC
system_prompt = """You are an expert on AMC and GPMC regulations. Provide:
1. Clear, concise answers with relevant citations from GPMC Act and AMC regulations
2. Step-by-step procedures when needed
3. References in [Source: GPMC Act/AMC Regulation, Section X] format
4. Important deadlines and compliance requirements
Structure responses with headers and bullet points as needed."""

# Initialize clients
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("amcbots")

def chunk_text(text: str, max_tokens: int = 1000) -> List[str]:
    """Split text into smaller chunks with stricter token limits."""
    chars_per_chunk = max_tokens * 4
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > chars_per_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def get_embedding(text: str, max_retries: int = 3) -> List[float]:
    """Get embedding with retry logic."""
    for attempt in range(max_retries):
        try:
            if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in text):
                text = translate_text(text, 'en')
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)

def summarize_context(context: str, max_tokens: int = 800) -> str:
    """Summarize long context to fit within token limits."""
    try:
        messages = [
            {"role": "system", "content": "Summarize the following text concisely while preserving key information:"},
            {"role": "user", "content": context}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        if DEBUG:
            st.error(f"Summarization error: {str(e)}")
        return context[:max_tokens * 4]

def search_pinecone(query: str, k: int = 2):
    """Search Pinecone with reduced results."""
    try:
        query_embedding = get_embedding(query)
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        return results
    except Exception as e:
        if DEBUG:
            st.error(f"Pinecone search error: {str(e)}")
        return None

def translate_text(text: str, target_lang: str, chunk_size: int = 3000) -> str:
    """Improved translation with chunking."""
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)

        if len(text) <= chunk_size:
            return translator.translate(text)

        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        translated_chunks = []

        for chunk in chunks:
            translated_chunk = translator.translate(chunk)
            translated_chunks.append(translated_chunk)
            time.sleep(0.5)

        return ' '.join(translated_chunks)
    except Exception as e:
        if DEBUG:
            st.error(f"Translation error: {str(e)}")
        return text

def generate_response(query: str, context: str, system_prompt: str):
    """Generate response with stricter context handling."""
    try:
        # Summarize context if too long
        context_tokens = len(context) // 4
        if context_tokens > 1000:
            context = summarize_context(context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context[:4000]}\n\nQuestion: {query[:500]}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content
    except Exception as e:
        if DEBUG:
            st.error(f"Response generation error: {str(e)}")
        return "Sorry, there was an error generating the response. Please try again."

# Streamlit UI
st.title("AMC GPMC Act Assistant")
st.write("Ask any question about the Gujarat Provincial Municipal Corporations Act and AMC regulations")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("What would you like t
