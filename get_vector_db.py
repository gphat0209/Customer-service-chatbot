from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# base_dir = "diva_data/facility"
# stored_db = "db/facility_db"
# all_file_paths = []

os.environ["GOOGLE_API_KEY"] = ""


# for root, dirs, files in os.walk(base_dir):
#     for file in files:
#         file_path = os.path.join(root, file)
#         all_file_paths.append(file_path)

# embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
# if not os.path.exists(stored_db):
#     os.makedirs(stored_db)

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

vectorstore = Chroma(persist_directory=stored_db, embedding_function=embedding)
