from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mysite.model_workflow import main_workflow

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WorkflowResponse(BaseModel):
    workflow_response: dict = Field(description="Workflow response")


@app.get("/scraping", response_model=WorkflowResponse)
async def model_ocr(url: str):
    print('INPUT URL:', url)
    response = main_workflow(url)
    return {
        "workflow_response": response
    }


#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)