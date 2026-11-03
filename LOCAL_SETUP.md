# FinSight - Local Development Setup

> **Note:** These instructions are for local development and smoke testing using Ollama. 

## Prerequisites & Installation

1. **Download and Install Ollama**
   * Get the installer from [https://ollama.com/download](https://ollama.com/download).

2. **Start Ollama**
   * Launch the Ollama service to ensure it is running locally at `http://localhost:11434`.

3. **Pull Required Models**
   * Run the following commands in your terminal to download the models specified in `.env.example`:
     ```bash
     ollama pull llama3.2:3b
     ollama pull nomic-embed-text
     ```

4. **Install Python Dependencies**
   * Install the required packages in your active virtual environment:
     ```bash
     pip install -r requirements.txt
     ```

5. **Configure Environment Variables**
   * Create your local `.env` file using `.env.example` as a template:
     ```bash
     cp .env.example .env
     ```

## Running the Smoke Test

Once your environment is configured and Ollama is running, execute the smoke test to verify all components (async chat, structured JSON, and embeddings):

```bash
python scripts/smoke_llm.py