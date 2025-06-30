import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from autogen import AssistantAgent, UserProxyAgent

from langchain.utilities import SerpAPIWrapper

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
embeddings = OpenAIEmbeddings()

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

st.title("Agentic RAG: Local PDF + Web Search")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)

    if len(chunks) == 0:
        st.error("No text found in PDF. Try a different file.")
    else:
        st.success(f"Loaded {len(chunks)} chunks.")

        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        retriever = vector_store.as_retriever()

        local_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )

        # Local RAG agent
        rag_agent = AssistantAgent(
            name="LocalRAG",
            llm_config={"config_list": [{"model": "gpt-4o"}]},
        )

        # Web search agent (LangChain SerpAPI wrapper)
        serp = SerpAPIWrapper()
        web_search_agent = AssistantAgent(
            name="WebSearch",
            llm_config={"config_list": [{"model": "gpt-4o"}]},
        )

        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        query = st.text_input("Ask a question about your PDF or the web:")

        if query:
            st.info("Agent is thinking...")

            local_answer = local_qa.run(query)

            if len(local_answer.strip()) < 20:
                serp_result = serp.run(query)
                final_answer = f"**Web Search:** {serp_result}"
            else:
                final_answer = f"**Local PDF Answer:** {local_answer}"

            st.subheader("Answer")
            st.write(final_answer)
