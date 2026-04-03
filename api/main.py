from fastapi import FastAPI
from pydantic import BaseModel

from agents.graph import graph

from fastapi import HTTPException
from services.contact_schema import ContactRequest
from services.mail_service import send_contact_email

app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str

# Response schema
class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    result = graph.invoke({
        "user_input": request.query,
        "intent": "",
        "response": ""
    })

    return {
        "answer": result["response"],
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/contact")
async def contact(data: ContactRequest):
    try:
        await send_contact_email(
            name=data.name,
            email=data.email,
            message=data.message
        )
        return {"status": "success"}
    except Exception as e:
        
        raise HTTPException(status_code=500, detail="Failed to send message")
