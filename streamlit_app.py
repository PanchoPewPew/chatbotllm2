import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

from openai import OpenAI

# App Title and Description
st.title("💬 Chatbot with RAG")
st.write(
    "This is a chatbot using a free GPT model (`gpt-neo`) and Retrieval-Augmented Generation (RAG) to provide answers from uploaded documents."
)

# PDF Upload Section
uploaded_file = st.file_uploader("Upload a PDF document for the chatbot to learn from:", type="pdf")

if uploaded_file:
    # Parse PDF Content
    from PyPDF2 import PdfReader
    reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()

    st.success("PDF uploaded and processed successfully!")

    # Initialize Embeddings and Vectorstore
    st.write("Creating embeddings for the uploaded document...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts([pdf_text], embeddings)

    # Set up the Free GPT Model Pipeline
    st.write("Initializing GPT model...")
    gpt_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    llm = HuggingFacePipeline(pipeline=gpt_pipeline)

    # Set up Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

    # Session State for Conversation
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display Chat Messages
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Field for User Messages
    if prompt := st.chat_input("Ask something based on the document!"):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a Response from the Chatbot
        response = qa_chain.run({"question": prompt, "chat_history": st.session_state.history})
        with st.chat_message("assistant"):
            st.markdown(response)

        # Save Assistant Response to History
        st.session_state.history.append({"role": "assistant", "content": response})