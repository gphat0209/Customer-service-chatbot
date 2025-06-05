from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import json

os.environ["GOOGLE_API_KEY"] = ""

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

json_files = ["contact.json", "ve-chung-toi.json", "diva_data/facility/lien-he.json", "diva_data/human_resources/human.json"]

merged_data = []

for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            print(f"File {file} is not JSON!")

with open("merged_data.json", "w", encoding="utf-8") as f_out: #merged data
    json.dump(merged_data, f_out, ensure_ascii=False, indent=2)

docs = []
loader = TextLoader("merged_data.json", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs.extend(text_splitter.split_documents(documents))
vectorstore = Chroma.from_documents(docs, embedding, persist_directory="db/general_info_db") #save merged data as db
