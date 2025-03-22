import os
import argparse
from typing import Optional

import google.generativeai as genai

class SimpleGeminiQA:
    """Simple Question Answering using Gemini's search capabilities."""

    def __init__(self, 
                 api_key=None,
                 model_name="gemini-1.5-flash",
                 temperature=0.2,
                 max_output_tokens=500):
        """
        Initialize the Simple Gemini QA system.
        
        Args:
            api_key: Gemini API key (if None, will try to get from GEMINI_API_KEY env var)
            model_name: The Gemini model to use
            temperature: Generation temperature
            max_output_tokens: Maximum output tokens for the model
        """
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        elif os.environ.get("GEMINI_API_KEY"):
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        else:
            raise ValueError("Gemini API key must be provided either directly or via GEMINI_API_KEY environment variable")
        
        # Configure generation parameters
        self.generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "text/plain",
        }
        
        # Create the model with search tools
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            tools=[
                genai.protos.Tool(
                    google_search_retrieval=genai.protos.GoogleSearchRetrieval(
                        dynamic_retrieval_config=genai.protos.DynamicRetrievalConfig(
                            mode=genai.protos.DynamicRetrievalConfig.Mode.MODE_DYNAMIC,
                            dynamic_threshold=0.1,
                        ),
                    ),
                ),
            ],
        )
        
        # Start a chat session
        self.chat = self.model.start_chat(history=[])
        
    def ask(self, question: str) -> str:
        """
        Ask a question and get a direct answer with search-based knowledge.
        
        Args:
            question: The question to answer
            
        Returns:
            A string containing the answer
        """
        try:
            # Send the question to the model
            response = self.chat.send_message(question)
            
            # Return the text response
            return response.text.strip()
                
        except Exception as e:
            return f"Error getting answer: {str(e)}"


def main():
    """Command line interface for the Simple Gemini QA."""
    parser = argparse.ArgumentParser(description="Simple Gemini QA Tool")
    
    parser.add_argument("--question", type=str, help="Question to ask")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Create the QA system
    qa = SimpleGeminiQA(
        api_key=args.api_key,
        model_name=args.model,
        temperature=args.temperature
    )
    
    if args.interactive:
        print("\n=== 🔍 Simple Gemini QA - Interactive Mode 🔍 ===")
        print("Type 'exit', 'quit', or Ctrl+C to exit")
        print("---------------------------------------------")
        
        while True:
            try:
                question = input("\n질문: ")
                if question.lower() in ['exit', 'quit']:
                    break
                    
                print("\n답변:")
                answer = qa.ask(question)
                print(answer)
                print("\n---------------------------------------------")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
    elif args.question:
        answer = qa.ask(args.question)
        print(answer)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()