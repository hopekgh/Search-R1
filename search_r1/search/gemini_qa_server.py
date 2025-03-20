import os
import json
import argparse
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .simple_gemini_qa import SimpleGeminiQA

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Launch Gemini QA server.")
parser.add_argument("--api_key", type=str, default=None, help="Gemini API key")
parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Name of the Gemini model.")
parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation.")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
args = parser.parse_args()

# Define request and response models
class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 1  # Not actually used since we only return one answer
    return_scores: bool = False

# Initialize the FastAPI app
app = FastAPI(title="Gemini QA API", description="API for answering questions using Gemini's search capabilities")

# Create a global QA instance
qa_system = SimpleGeminiQA(
    api_key=args.api_key,
    model_name=args.model, 
    temperature=args.temperature
)

@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and returns answers using Gemini's search capabilities.
    
    Input format:
    {
      "queries": ["윤석열 탄핵심판 어떻게 됐어?", "Another question?"],
      "topk": 1,
      "return_scores": false
    }
    """
    if not request.queries:
        raise HTTPException(status_code=400, detail="No queries provided")
    
    try:
        # Process each query and get answers
        results = []
        
        for query in request.queries:
            # Get answer from Gemini
            answer = qa_system.ask(query)
            
            # Return single result with the answer directly in search result format
            # This is the simplest format that will work with the existing generation.py
            if request.return_scores:
                # If scores requested, wrap in a format expected by _passages2string
                results.append([{"document": {"contents": answer}, "score": 1.0}])
            else:
                # Simple format without scores
                results.append([{"contents": answer}])
                
        return {"result": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint to verify the server is running."""
    return {"status": "healthy", "service": "Gemini QA API"}

if __name__ == "__main__":
    # Launch the server
    print(f"Starting Gemini QA API server with model: {args.model}, temperature: {args.temperature}")
    print(f"Server will be available at http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)