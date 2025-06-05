from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import logging
import time
from datetime import datetime


os.environ["GOOGLE_API_KEY"] = ""

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
# vectorstore = Chroma(persist_directory="db/cham-soc-da_db", embedding_function=embedding)
# retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

chat_template = """Bạn là một chatbot đại diện cho thẩm mỹ viện Diva để giải đáp thông tin, thắc mắc của khách hàng. 
Sử dụng thông tin được cung cấp của thẩm mỹ viện Diva để trả lời các câu hỏi từ khách hàng. 
Nếu bạn không biết câu trả lời, cứ trả lời ràng bạn không biết. 
Trả lời trong tối đa 3 câu và câu trả lời nên xúc tích nhất có thể.
Câu hỏi: {question} 
Thông tin: {context} 
Câu trả lời:
"""

prompt = ChatPromptTemplate.from_template(chat_template)

def get_retriever(db_path):
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding)
    retriever = vectorstore.as_retriever()
    return retriever

def build_rag_chain(retriever):
    return (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

class ChatInput(BaseModel):
    user_input: str


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f"{session_id}.log")


logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    filemode="w",
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI()
session_state = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory='templates')

@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse('chat.html', {'request': request})


@app.post("/chatbot")
def qa_chatbot(request: Request, input: ChatInput):
    user_id = "default_user"  
    state = session_state.get(user_id, {"step": 0, "retriever": None})
    # logger.info(f"Người dùng hỏi: {user_input} | State: {state}")

    try:
        if state["step"] == 0:
            db_map = {
                "1": "db/general_info_db",
                "2": "services_db",
                "3": "db/beauty_db"
            }
            if input.user_input.startswith("1"):
                db_dir = db_map.get(input.user_input.strip())
                retriever = get_retriever(db_dir) 
                session_state[user_id] = {"step":2, "retriever":retriever}
                logger.info(f"Người dùng yêu cầu: {db_map.get(db_dir)} | State: {state}")
                return {"results": "Bạn muốn biết thêm gì về chúng tôi? "}
                # results = rag_chain.invoke(input.user_input, retriever=retriever)
                # return {"results": results}

            elif input.user_input.startswith("2"):
                session_state[user_id] = {"step": 1}
                return {"results": "Bạn muốn tư vấn về gì?\n1. Chăm sóc da\n2. Điều trị da\n3. Phun xăm\n4. Thẩm mỹ công nghệ cao"}

            elif input.user_input.startswith("3"):
                db_dir = db_map.get(input.user_input.strip())
                retriever = get_retriever(db_dir) 
                session_state[user_id] = {"step":2, "retriever":retriever}
                logger.info(f"Người dùng yêu cầu: {db_map.get(db_dir)} | State: {state}")
                return {"results": "Bạn muốn biết thêm gì về kiến thức làm đẹp?"}
            else:
                return {"results": "Vui lòng chọn 1, 2 hoặc 3."}
            
        elif state["step"] == 1:
            db_map = {
                "1": "db/cham-soc-da_db",
                "2": "db/dieu-tri-da_db",
                "3": "db/phun-xam-tham-my_db",
                "4": "db/tham-my-cong-nghe-cao_db"
            }
            db_dir = db_map.get(input.user_input.strip())
            if not db_dir:
                return {"results": "Vui lòng chọn 1 đến 4 cho loại dịch vụ thẩm mỹ."}
            logger.info(f"Người dùng yêu cầu: {db_dir} | State: {state}")
            retriever = get_retriever(db_dir) 
            # rag_chain = build_rag_chain(retriever)
            session_state[user_id] = {"step": 2, "retriever": retriever}
            # results = rag_chain.invoke(input.user_input, retriever=retriever)
            return {"results": "Nhập câu hỏi của bạn"}
        elif state["step"] == 2:
            retriever = state["retriever"]
            rag_chain = build_rag_chain(retriever)

            start_time = time.time()
            results = rag_chain.invoke(input.user_input)
            elapsed = time.time() - start_time

            logger.info(f"Người dùng hỏi: {input.user_input} | State: {state} | Chatbot trả lời: {results} | Thời gian xử lý: {elapsed:.2f} giây" )
            return {"results": results}
        else:
            session_state[user_id] = {"step": 0}
            return {"results": "Xin lỗi, tôi chưa hiểu bạn cần gì. Vui lòng chọn 1, 2 hoặc 3."}

    except Exception as e:
        logger.error(f"Lỗi xảy ra: {str(e)}", exc_info=True)
        return {"results": "Đã xảy ra lỗi trong quá trình xử lý câu hỏi của bạn."}
    
@app.post("/reset_session")
def reset_session():
    user_id = "default_user"
    session_state[user_id] = {"step": 0}
    return {"message": "Session reset"}


    
