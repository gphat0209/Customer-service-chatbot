from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.chains import RetrievalQA
import os

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import logging
import time


os.environ["GOOGLE_API_KEY"] = ""

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
vectorstore = Chroma(persist_directory="db/cham-soc-da_db", embedding_function=embedding)
retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

chat_template = """Bạn là một chatbot để giải đáp thông tin, thắc mắc của khách hàng về thẩm mỹ viện Diva. 
Sử dụng thông tin được cung cấp để trả lời các câu hỏi từ khác hàng. 
Nếu bạn không biết câu trả lời, cứ trả lời ràng bạn không biết. 
Trả lời trong tối đa 3 câu và câu trả lời nên xúc tích nhất có thể
Câu hỏi: {question} 
Thông tin: {context} 
Câu trả lời:
"""

prompt = ChatPromptTemplate.from_template(chat_template)

def build_rag_chain(retriever):
    return (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

rag_chain = build_rag_chain(retriever)

# rag_chain = (
#     {"context": retriever,  "question": RunnablePassthrough()} 
#     | prompt 
#     | llm
#     | StrOutputParser() 
# )

app = FastAPI()
templates = Jinja2Templates(directory='templates')

class ChatInput(BaseModel):
    user_input: str

logging.basicConfig(
    level=logging.INFO,
    filename="chatbot_logs.log",
    filemode="a",  # Ghi tiếp vào file cũ
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
# from utils import load_document_store

@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse('chat.html', {'request': request})

@app.post("/chatbot")
def qa_chatbot(input: ChatInput):
    results = rag_chain.invoke(input.user_input)
    return {"results": results}

@app.post("/chatbot")
def qa_chatbot(input: ChatInput):
    all_results = []
    results = rag_chain.invoke(input.user_input)
    return {"results": results}


# @app.post("/chatbot")
# def qa_chatbot(input: ChatInput):
#     logger.info(f"Người dùng hỏi: {input.user_input}")

#     try:
#         # Truy vấn context từ retriever
#         # context_docs = retriever.get_relevant_documents(input.user_input)
#         # logger.info(f"Context trả về ({len(context_docs)} documents)")

#         # Trả lời từ mô hình
#         start = time.time()
#         results = rag_chain.invoke(input.user_input)
#         elapsed = time.time() - start.time()
#         logger.info(f"Thời gian xử lý: {elapsed:.2f} giây")
#         logger.info(f"Phản hồi từ chatbot: {results}")

#         return {"results": results}

#     except Exception as e:
#         logger.error(f"Lỗi xảy ra: {str(e)}", exc_info=True)
#         return {"results": "Đã xảy ra lỗi trong quá trình xử lý câu hỏi của bạn."}
    
