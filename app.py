import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from typing import List
from deep_translator import GoogleTranslator

# System prompt definition
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

For every fact or statement, include a reference to the source document and page number in this format:
[Source: Document_Name, Page X]

Always structure your responses in a clear, organized manner using:
- Bullet points for lists
- Numbered steps for procedures
- Bold text for important points
- Separate sections with clear headings"""

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])

def get_embedding(text: str) -> List[float]:
    """Get embedding for the input text using OpenAI's embedding model."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def search_pinecone(query: str, k: int = 3):
    """Search Pinecone index with embedded query."""
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    return results

def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language using deep-translator."""
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        return f"Translation Error: {str(e)}"

def generate_response(query: str, context: str, system_prompt: str):
    """Generate response using OpenAI with context and system prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content

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

# Chat input and processing
if prompt := st.chat_input("What would you like to know?"):
    # Check if input is in Gujarati and translate to English if needed
    if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in prompt):
        english_prompt = translate_text(prompt, 'en')
        # Display original Gujarati query and its English translation
        with st.chat_message("user"):
            st.markdown(f"Original Query (ગુજરાતી): {prompt}")
            st.markdown(f"Translated Query (English): {english_prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        prompt = english_prompt
    else:
        # Display original English query
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Search Pinecone and get relevant context
    search_results = search_pinecone(prompt)
    context = "\n".join([result.metadata.get('text', '') for result in search_results.matches])

    # Generate response
    with st.chat_message("assistant"):
        response = generate_response(prompt, context, system_prompt)

        # Create columns for response and translation button
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(response)

        with col2:
            if st.button("ગુજરાતી 🔄", key=f"translate_{len(st.session_state.messages)}"):
                gujarati_response = translate_text(response, 'gu')
                st.markdown("ગુજરાતી અનુવાદ (Gujarati Translation):")
                st.markdown(gujarati_response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.write("This chatbot provides information about the Gujarat Municipal Act and Ahmedabad Municipal Corporation.")
    st.write("It uses AI to search through the official documentation and provide accurate responses with references.")
    st.write("""
    Language Features:
    - You can ask questions in English or Gujarati
    - If you ask in Gujarati, it will be automatically translated to English
    - Click the 'ગુજરાતી 🔄' button to see responses in Gujarati
    """)
