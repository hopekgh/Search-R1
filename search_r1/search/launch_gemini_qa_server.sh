#!/bin/bash
# Launch script for Gemini QA server

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
  echo "Error: GEMINI_API_KEY environment variable is not set."
  echo "Please set it using: export GEMINI_API_KEY=your_key_here"
  exit 1
fi

# Default values
MODEL="gemini-1.5-flash"
TEMPERATURE=0.2
PORT=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "Starting Gemini QA Server..."
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Port: $PORT"

# Launch the server with parsed arguments
python -m search_r1.search.gemini_qa_server \
  --model "$MODEL" \
  --temperature "$TEMPERATURE" \
  --port "$PORT"