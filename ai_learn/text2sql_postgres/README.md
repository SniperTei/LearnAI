# PostgreSQL Text-to-SQL Demo

This is a Streamlit-based Text-to-SQL application that uses PostgreSQL as the backend database instead of SQLite. The application allows users to query a PostgreSQL database using natural language.

## Setup Instructions

1. Install PostgreSQL on your system if not already installed
2. Create a PostgreSQL database and user for this application
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your PostgreSQL credentials and OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=your_database_name
   POSTGRES_USER=your_username
   POSTGRES_PASSWORD=your_password
   ```
5. Run the database initialization script to create tables and populate sample data:
   ```bash
   python init_database.py
   ```
6. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Features

- Natural language to SQL conversion using OpenAI's GPT-4o model
- PostgreSQL database integration
- Interactive web interface built with Streamlit
- Query result visualization and download options
- Example queries to help users get started

## Database Schema

The application uses an insurance-themed database with the following tables:
- `customers`: Customer information
- `products`: Insurance products
- `policies`: Policy information linking customers and products
- `claims`: Claims filed against policies