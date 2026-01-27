from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SolProd AI Assistant")

# CORS для вашего фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://solprod.agency",
        "https://solprod-ai.onrender.com",
        "http://localhost:3000",
        "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-exp',
    system_instruction="You are a helpful AI assistant for SolProd agency. SolProd is a full-cycle team that unites experts in design, development, and production. Help users with their questions about services, portfolio, and projects. Be professional and friendly."
)

class ChatMessage(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    message: str
    conversation_history: list

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(data: ChatMessage):
    try:
        # Конвертируем историю в формат Gemini
        gemini_history = []
        for msg in data.conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        # Создаем чат с историей
        chat_session = model.start_chat(history=gemini_history)
        
        # Отправляем сообщение
        response = chat_session.send_message(data.message)
        assistant_message = response.text
        

        conversation = data.conversation_history + [
            {"role": "user", "content": data.message},
            {"role": "assistant", "content": assistant_message}
        ]
        
        return ChatResponse(
            message=assistant_message,
            conversation_history=conversation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)