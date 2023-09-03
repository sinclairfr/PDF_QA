from dotenv import load_dotenv
import openai
import os
import time
import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    
    # streamlit
    st.set_page_config(page_title="PDF Q&A ❓")
    st.header("PDF Q&A")
    pdf = st.file_uploader("Upload your pdf file 🗄", type="pdf")
    
    if  pdf is not None:
        pdf_reader = PdfReader(pdf)     
        text = ""
        # reading pdf
        for pages in pdf_reader.pages:
            text += pages.extract_text()
        #st.text(text)
        
        # Initializing the text splitter
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        # getting the chunks
        chunks = text_splitter.split_text(text)
        
        # progress bar
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0)
        total_chunks = len(chunks)
    
        # embeddings
        embeddings = OpenAIEmbeddings()
        for index, chunk in enumerate(chunks):
            # This is where you'd calculate embeddings for each chunk
            # For demonstration purposes, I'm just sleeping for a short duration
            time.sleep(0.1)
            
            # Update the progress bar
            progress = (index + 1) / total_chunks
            my_bar.progress(progress)

        knowledge_base = FAISS.from_texts(chunks, embeddings)
                
        # user question
        user_question = st.text_input('Ask a question about your PDF.')
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = user_question)
                st.write("🎯" + response)
                st.warning(cb)

if __name__ == "__main__":
    main()