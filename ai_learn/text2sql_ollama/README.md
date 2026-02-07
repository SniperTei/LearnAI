# Ollama Text-to-SQL Demo

This is a Text-to-SQL application that uses Ollama as the language model instead of OpenAI. The application allows users to query a SQLite database using natural language.

## Prerequisites

1. **Ollama** must be installed and running on your system
2. At least one Ollama model should be pulled (e.g., `llama3`, `mistral`, `codellama`)

## Setup Instructions

1. Make sure Ollama is running on your system:
   ```bash
   ollama serve
   ```

2. Pull a model (if not already done):
   ```bash
   ollama pull llama3
   # or
   ollama pull mistral
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Features

- Natural language to SQL conversion using Ollama
- SQLite database integration
- Interactive web interface built with Streamlit
- Query result visualization and download options
- Example queries to help users get started
- Model selection from available Ollama models

## Database Schema

The application uses an insurance-themed database with the following tables:
- `customers`: Customer information
- `products`: Insurance products
- `policies`: Policy information linking customers and products
- `claims`: Claims filed against policies

## Available Models

The application supports various Ollama models:
- llama3 (recommended)
- mistral
- codellama
- phi

You can select your preferred model from the sidebar when using the application.