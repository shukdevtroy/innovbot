from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS #facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import concurrent.futures

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask InnovBot")
    
    # Add logo image
    logo_path = 'Innovation4.png'
    st.image(logo_path, width=200)
    
    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    st.header("Ask InnovBot")
    
    # Provide the file path to the PDF file
    pdf_path = 'QA dataset.pdf'
    
    # Open the PDF file
    with open(pdf_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the text into chunks
        chunk_size = 1000
        chunk_overlap = 200
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

        # Create embedding
        embeddings = HuggingFaceEmbeddings()

        # Create knowledge base
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask Question:")
        if user_question:
            # Search for similar documents
            docs = knowledge_base.similarity_search(user_question)

            # Load question answering chain
            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":5, "max_length":500})
            chain = load_qa_chain(llm, chain_type="stuff")

            # Process the first chunk only
            response = process_chunk(docs[0], chain, user_question)

            # Display response
            st.write(response)

def process_chunk(doc, chain, user_question):
    # Process chunk
    response = chain.run(input_documents=[doc], question=user_question)

    return response

if __name__ == '__main__':
    main()
