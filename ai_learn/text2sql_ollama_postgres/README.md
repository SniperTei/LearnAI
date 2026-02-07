# Ollama PostgreSQL Text-to-SQL Demo

This is a Text-to-SQL application that uses Ollama as the language model with PostgreSQL as the database backend. The application allows users to query a PostgreSQL database using natural language.

## Prerequisites

1. **Ollama** must be installed and running on your system
2. At least one Ollama model should be pulled (e.g., `llama3`, `mistral`, `codellama`)
3. **PostgreSQL** must be installed and running on your system

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

3. Make sure PostgreSQL is installed and running

4. Create a PostgreSQL database and user for this application

5. Create a `.env` file with your PostgreSQL credentials:
   ```
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=your_database_name
   POSTGRES_USER=your_username
   POSTGRES_PASSWORD=your_password
   ```

6. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

7. Run the database initialization script to create tables and populate sample data:
   ```bash
   python init_database.py
   ```

8. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Features

- Natural language to SQL conversion using Ollama
- PostgreSQL database integration
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