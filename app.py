from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import gradio as gr
import os

key = os.getenv('key')

def pdf_file_reader(doc,question):
  pdf_docs = PyPDFLoader(doc,extract_images=True).load()

  #pdf_docs = OnlinePDFLoader(doc).load()

  llm = ChatGroq(temperature=0, model_name="Llama3-8b-8192",groq_api_key=key)

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  #The overlap helps mitigate the possibility of separating a statement from important context related to it.

  splits = text_splitter.split_documents(pdf_docs)
  # Split Documents and return type is List[Document]

  vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())

  # Retrieve and generate using the relevant snippets of the blog.
  retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) #search_kwargs={"k": 1}. vectorstore.similarity_search_with_score(query)

  template = """Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  Use three sentences maximum and keep the answer as concise as possible.
  Always say "thanks for asking!" at the end of the answer.
  {context}
  Question: {question}
  Helpful Answer: """
  custom_rag_prompt = PromptTemplate.from_template(template)

  rag_chain = (
      {"context": retriever , "question": RunnablePassthrough()}
      | custom_rag_prompt
      | llm
      | StrOutputParser()
  )

  return rag_chain.invoke(question)


# Set up the Gradio interface
demo = gr.Interface(
    fn=pdf_file_reader,
    inputs=["file", "text"],
    outputs=["text"],
    #outputs=["text"],
    title="Q/A",
    #outputs=["text"],
    #live=False,
    description="Upload a PDF file and ask a question!",
)

demo.launch(debug=True)
