# 🦜️🔗 PDF Questions and Answers on your PDF with LangChain

This repo is an implementation of a locally hosted chatbot specifically focused on question answering on your onw PDF. Small app built following the [LangChain documentation](https://langchain.readthedocs.io/en/latest/).

Deployed version: [chat.langchain.com](https://chat.langchain.com)

## ✅ Running locally
1. Install backend dependencies: `pip install -r requirements.txt`.
2. Rename the `.env.example` to `.env` 
3. Enter your OpenAI API key in the environment variables in the `.env` file:
```
OPENAI_API_KEY="sk-***"
```
4. Run `streamlit main.py` 
5. Open [localhost:3000](http://localhost:3000) in your browser.

## 📚 Technical description

The application reads the PDF and splits the text into smaller chunks that can be then fed into a LLM. It uses OpenAI embeddings to create vector representations of the chunks. The application then finds the chunks that are semantically similar to the question that the user asked and feeds those chunks to the LLM to generate a response.

The application uses Streamlit to create the GUI and Langchain to deal with the LLM.

There are two components: 
- embeddings
- question-answering.

![process](images/process.jpg)

Source : [Benny Cheung](https://bennycheung.github.io/ask-a-book-questions-with-langchain-openai)

Question-Answering has the following steps, all handled by [OpenAIFunctionsAgent](https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent):

1. Given the user input question, determine what a standalone question would be (using GPT-3.5).
2. Given that standalone question, look up relevant documents chunks.
3. Pass the standalone question and relevant document chunks to GPT to generate and stream the final answer.

## 🚀 Deployment

TODO : docker deployment


## Contributing

This repository is for educational purposes only and is not intended to receive further contributions. It is supposed to be used as support material for the YouTube tutorial that shows how to build the project.


## Blog Posts:
* [Article](www.domainelibre.com)
