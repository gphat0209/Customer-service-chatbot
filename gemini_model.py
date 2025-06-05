from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
# from langchain.chains import RetrievalQA
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser

os.environ["GOOGLE_API_KEY"] = ""


# base_dir = "diva_data/facility"
# all_file_paths = []

# for root, dirs, files in os.walk(base_dir):
#     for file in files:
#         file_path = os.path.join(root, file)
#         all_file_paths.append(file_path)


# for file in all_file_paths:
#     loader = TextLoader(file, encoding="utf-8")
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs.extend(text_splitter.split_documents(documents))

stored_db = "services_db"
embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
# docs = []

for file in os.listdir("diva_data/services"):    
    docs = []
    loader = TextLoader(f"diva_data/services/{file}", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs.extend(text_splitter.split_documents(documents))
    filename = file.split(".json")[0]
    vectorstore = Chroma.from_documents(docs, embedding, persist_directory=filename + "_db")


# if not os.path.exists(stored_db):
#     vectorstore = Chroma.from_documents(docs, embedding, persist_directory=stored_db)

# vectorstore = Chroma(persist_directory=stored_db, embedding_function=embedding)
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)


# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")



# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,  # retriever từ vectorstore
#     return_source_documents=True,
# )

# query = "Nguyễn Thành Trí là bác sĩ gì?"
# result = qa_chain.invoke({"query": query})

# print("Câu trả lời:", result["result"])
