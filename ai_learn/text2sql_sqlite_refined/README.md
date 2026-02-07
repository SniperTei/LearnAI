# Refined SQLite Text-to-SQL Demo

This is a refined version of the Text-to-SQL application that uses SQLite as the backend database. The application allows users to query a SQLite database using natural language.

## Setup Instructions

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. Run the database initialization script to create tables and populate sample data:
   ```bash
   python init_database.py
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Features

- Natural language to SQL conversion using OpenAI's GPT-4o model
- SQLite database integration
- Interactive web interface built with Streamlit
- Query result visualization and download options
- Example queries to help users get started

## Database Schema

The application uses an insurance-themed database with the following tables:
- `customers`: Customer information
- `products`: Insurance products
- `policies`: Policy information linking customers and products
- `claims`: Claims filed against policies