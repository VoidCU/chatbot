from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.document_loaders import TextLoader

# Initialize LLM and parser
ollama_llm = OllamaLLM(model="llama3")
parser = StrOutputParser()

# Load documents
loader = TextLoader('data.txt', encoding='utf-8')
document = loader.load()

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
pages = splitter.split_documents(document)

# Create FAISS vector store from documents
vector_storage = FAISS.from_documents(pages, embedding=OllamaEmbeddings(model='llama3'))
retriever = vector_storage.as_retriever()

# Define prompt template
template = """
You are an AI-powered chatbot designed to assist visitors of our university website. Your role is to provide accurate and helpful information based on the provided context.
Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create the chain
result = RunnableParallel(context=retriever, question=RunnablePassthrough())
chain = result | prompt | ollama_llm | parser

# Example question
response = chain.invoke({'question': 'What are the application requirements for undergraduate programs?'})
print(response)
