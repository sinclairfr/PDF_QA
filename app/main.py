# Standard libraries
import os
import time

# External libraries
from dotenv import load_dotenv
import openai
import streamlit as st
from PyPDF2 import PdfReader

# Custom/local modules
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    # Load environment variables
    load_dotenv()

    # Streamlit UI setup
    st.set_page_config(page_title="PDF Q&A ❓")
    st.header("PDF Questions & Answers")
    pdf = st.file_uploader("Upload your pdf file 🗄", type="pdf")

    # Process the uploaded PDF
    if pdf is None:
        st.info("Please upload your pdf file.")
    else:
        pdf_reader = PdfReader(pdf)
        text = ''.join(page.extract_text() for page in pdf_reader.pages)

        # Split the extracted text into manageable chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Display progress bar while processing chunks
        my_bar = st.progress(0)
        total_chunks = len(chunks)

        embeddings = OpenAIEmbeddings()
        for index, chunk in enumerate(chunks):
            # Placeholder for actual processing - using sleep for demonstration
            time.sleep(0.1)
            my_bar.progress((index + 1) / total_chunks)

        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Get user's question and search for answers in the knowledge base
        user_question = st.text_input('Ask a question about your PDF.')
        if not user_question:
            st.info("Please ask your question.")
        else:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                st.write("🎯" + response)
                st.warning(cb)

if __name__ == "__main__":
    main()