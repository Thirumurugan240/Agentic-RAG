import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.utilities import SerpAPIWrapper

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),model_name="gpt-4o", temperature=0.2)
embeddings = OpenAIEmbeddings()
serp = SerpAPIWrapper(serpapi_api_key=os.getenv("SERP_API_KEY"))

st.title("Agentic RAG: Local PDF + Web Search")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    chunks = text_splitter.split_text(raw_text)

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vector_store.as_retriever()
    local_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    st.success(f"PDF loaded and indexed ({len(chunks)} chunks)")

    config_list = [{"model": "gpt-4o"}]

    local_rag_agent = AssistantAgent(
        name="LocalRAG",
        llm_config={"config_list": config_list},
    )

    web_agent = AssistantAgent(
        name="WebSearch",
        llm_config={"config_list": config_list},
    )

    user = UserProxyAgent(
        name="User",
        code_execution_config={"use_docker": False},  # disables Docker runtime
    )


    query = st.text_input("Ask a question:")

    if query:
        st.info("Orchestrating agents...")

        # 1) Local answer
        local_answer = local_qa.run(query)

        # 2) Ask coordinator LLM to decide if more is needed
        coordinator_prompt = f"""
You are a smart AI assistant orchestrating Local RAG and Web Search.

Here is the answer from the local PDF:
\"\"\"{local_answer}\"\"\"

Decide if this answer is enough.
- If YES: return the local answer.
- If NO: do a web search for \"{query}\" and combine the results.

Provide a final clear answer.
"""

        # Actually ask the LLM to decide
        final_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
        # If local answer is not good, add web
        serp_result = serp.run(query)
        final_answer = final_llm.predict(
            f"""
PDF answer:
{local_answer}

Web search result:
{serp_result}

Based on both, write a single clear answer for the user.
"""
        )

        st.subheader("Final Answer")
        st.write(final_answer)
