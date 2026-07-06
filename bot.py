from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

# 2. Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0 
)

file_path = "club_policy.pdf"

# 3. Load and Split Document
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    add_start_index=True
)
chunks = text_splitter.split_documents(docs)

# 4. Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6}) #use 6 chunks as context

# 5. RAG Prompt Template
template = """
You are a precise assistant. You will be given a question, and 
will have to go through the given document, where all the answers
to the question would be given. Make sure you retrieve the proper
and correct procedure, formats, protocols, names, mail id, mail format, 
if relevant to the question. Don't miss important details. Only answer
what is asked.
If you don't find the answer in the context, say you don't know, 
and help them accordingly.

Context: {context}

Question: {question}

Detailed Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 6. RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Execution
query = "How does my club apply for putting digital standees?"
response = rag_chain.invoke(query)

print("\n Response:\n")
print(response)
