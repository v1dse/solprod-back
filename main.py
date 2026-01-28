from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SolProd AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://solprod.agency",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SYSTEM_INSTRUCTION = """You are an AI assistant for SolProd agency.

## About SolProd:
SolProd is a full-cycle team that unites experts in design, development, and production.
We've been in business since 2024 and have completed 10+ projects for clients worldwide.

## Contact Information:
- Email: solutions.production.manager@gmail.com
- Telegram: @ManagerSolProd
- Website: https://solprod.agency
- LinkedIn: linkedin.com/company/solprod
- Instagram: @sol.prod.team

## Services We Offer:
1. **Web Development:** 
   - Landing pages
   - Corporate websites
   - E-commerce platforms
   - Web applications
   
2. **Mobile Development:**
   - iOS apps
   - Android apps
   - Cross-platform solutions
   
3. **Design:**
   - UI/UX Design
   - Branding
   - Logo design
   - Marketing materials

4. **AI Solutions:**
   - Chatbots
   - AI integrations
   - Automation

## Pricing:
- Free consultation
- Custom quotes based on project scope
- Flexible payment plans available

## Process:
1. Initial consultation (free)
2. Project scope & quote
3. Design phase
4. Development
5. Testing & launch
6. Support & maintenance

## Portfolio:
Visit https://solprod.agency/#portfolio to see our work.

## Working Hours:
Monday - Friday: 9:00 - 18:00 (CET)
Saturday - Sunday: Closed
Emergency support available 24/7

## Languages:
We speak English, Polish, Russian, and Ukrainian.

## Instructions:
- Be professional, friendly, and helpful
- Provide accurate contact information
- Answer questions about services and pricing
- Suggest booking a free consultation for detailed discussions
- If you don't know something specific, recommend contacting the team
- Respond in the same language as the user
- Don't make up information - only use what's provided here
- Format lists using dashes (-) or numbers, NOT asterisks (*)
- Use clear, readable formatting without excessive markdown symbols
- Keep responses clean and professional


## Formatting Rules:
- Use simple, clean text formatting
- For bullet points, use dashes (-) or emojis (✓, •) instead of asterisks
- Don't use excessive markdown symbols
- Keep formatting minimal and readable
- Use line breaks for better readability
"""

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite',
    system_instruction=SYSTEM_INSTRUCTION  
)

class ChatMessage(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    message: str
    conversation_history: list

@app.get("/")
async def root():
    return {"status": "ok", "message": "SolProd AI API is running"}

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(data: ChatMessage):
    try:
        gemini_history = []
        for msg in data.conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        chat_session = model.start_chat(history=gemini_history)
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
        print(f"Error: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
