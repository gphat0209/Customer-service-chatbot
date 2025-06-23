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
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

import logging
import time
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableMap


os.environ["GOOGLE_API_KEY"] = ""

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
vectorstore = Chroma(persist_directory="db/merged2_db", embedding_function=embedding)
retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

chat_template = """Bạn là một chatbot đại diện cho thẩm mỹ viện Diva để giải đáp thông tin, thắc mắc của khách hàng. 
Sử dụng thông tin được cung cấp của thẩm mỹ viện Diva để trả lời các câu hỏi từ khách hàng. 
Bạn cũng nên dựa vào lịch sử trò chuyện được cung cấp để nâng cao trải nghiệm khách hàng.
Nếu bạn không biết câu trả lời, cứ trả lời ràng bạn không biết. 
Trả lời trong tối đa 3 câu và câu trả lời nên xúc tích nhất có thể.
Câu hỏi: {question} 
Lịch sử trò chuyện: {chat_history}
Thông tin: {context} 
Câu trả lời:
"""

# workflow = StateGraph(state_schema=MessagesState)

# # Gọi model, dùng thêm history
# def call_model(state: MessagesState):
#     messages = state["messages"]
#     latest_question = messages[-1].content

#     # Tạo lịch sử hội thoại dạng văn bản
#     history_text = ""
#     for m in messages[:-1]:  # exclude the current question
#         role = "Khách hàng" if isinstance(m, HumanMessage) else "Bot"
#         history_text += f"{role}: {m.content}\n"

#     # Tạo prompt template
#     prompt = ChatPromptTemplate.from_template(chat_template)

#     # Chain xử lý
#     chain = RunnableMap({
#         "context": lambda _: retriever.invoke(latest_question),
#         "question": lambda _: latest_question,
#         "history": lambda _: history_text.strip()
#     }) | prompt | llm | StrOutputParser() | (lambda output: {"messages": [AIMessage(content=output)]})

#     return chain

# # Thêm node & kết nối
# workflow.add_node("model", call_model)
# workflow.add_edge(START, "model")

# # Bộ nhớ để lưu hội thoại
# memory = MemorySaver()
# llm_app = workflow.compile(checkpointer=memory)

prompt = ChatPromptTemplate.from_template(chat_template)

chat_history = {}
chat_history["User"] = []
chat_history["AI Chatbot"] = []

def get_chat_history(input):
    return "\n".join([
        f"Khách hàng: {u}\nBot: {a}" for u, a in zip(chat_history["User"], chat_history["AI Chatbot"])
    ])
    return chat_history

def build_rag_chain(retriever):
    return (
    {"context": retriever,  "question": RunnablePassthrough(), "chat_history": RunnableLambda(get_chat_history)} 
    | prompt 
    | llm
    | StrOutputParser() 
)

rag_chain = build_rag_chain(retriever)

log_dir = "logs2"
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory='templates')

class ChatInput(BaseModel):
    user_input: str


@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse('chat.html', {'request': request})

@app.post("/chatbot")
def qa_chatbot(input: ChatInput):
    try:
        start_time = time.time()
        results = rag_chain.invoke(input.user_input)
        
        # results = llm_app.invoke({"messages": [HumanMessage(content=input.user_input)]}, config={"configurable": {"thread_id": "1"}})
        elapsed = time.time() - start_time

        if len(chat_history["User"]) == 10:
            chat_history["User"].pop(0)
            chat_history["AI Chatbot"].pop(0)
        
        chat_history["User"].append(input.user_input)
        chat_history["AI Chatbot"].append(results)

        logger.info(f"Người dùng hỏi: {input.user_input} | Chatbot trả lời: {results} | Thời gian xử lý: {elapsed:.2f} giây" )
        # return {"results": results["messages"][-1].content}
        return {"results":results}
    except Exception as e:
        logger.error(f"Lỗi xảy ra: {str(e)}", exc_info=True)
        return {"results": "Đã xảy ra lỗi trong quá trình xử lý câu hỏi của bạn."}



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
    
