# Streamlit UI
st.title("Gujarat Tax Law Assistant")
st.write("Ask any question about tax laws")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("What would you like to know?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner('Processing...'):
        # Translate to English if needed
        search_prompt = translate_text(prompt, 'en') if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in prompt) else prompt

        # Search and generate response
        search_results = search_pinecone(search_prompt)
        if search_results:
            # Limit context size more aggressively
            context = " ".join([result.metadata.get('text', '')[:2000]
                              for result in search_results.matches])
            response = generate_response(search_prompt, context, system_prompt)
        else:
            response = "Sorry, there is currently difficulty retrieving information. Please try again later."

        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This chatbot provides information about Gujarat Tax Laws and Ahmedabad Municipal Corporation.")
    st.write("""
    Language Features:
    - Questions in English or Gujarati
    - Responses in English
    """)
