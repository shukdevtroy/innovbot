from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS #facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

@st.cache
def main():
    load_dotenv()
    st.set_page_config(page_title="Ask InnovBot")
    # Add logo image
    logo_path = 'Innovation4.png'
    st.image(logo_path, width=200)

    # Add voice greeting
    voice_path = 'greeting.mp3'
    st.audio(voice_path, format='audio/mp3', start_time=0)

    st.header("Ask InnovBot")
    
    # Provide the file path to the PDF file
    pdf_path = 'QA dataset.pdf'
    
    # Open the PDF file
    with open(pdf_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # spilit ito chuncks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embedding
        embeddings = HuggingFaceEmbeddings()

        knowledge_base = FAISS.from_texts(chunks,embeddings)

        user_question = st.text_input("Ask Question:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":5,
                                                      "max_length":500})
            chain = load_qa_chain(llm,chain_type="stuff")
            response = chain.run(input_documents=docs,question=user_question)

            st.write(response)

if __name__ == '__main__':
    main()
