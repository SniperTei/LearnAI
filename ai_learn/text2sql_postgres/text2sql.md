# PostgreSQL Text-to-SQL Implementation Guide

This document explains the PostgreSQL implementation of the Text-to-SQL application, highlighting differences from the SQLite version and providing technical details.

## Key Differences from SQLite Version

### 1. Database Connection
- **SQLite**: Uses `sqlite3.connect()` with a file path
- **PostgreSQL**: Uses `psycopg2.connect()` with server connection parameters (host, port, database, username, password)

### 2. SQL Syntax Differences
While both databases use SQL, there are some syntax variations:
- **Auto-increment**: PostgreSQL uses `SERIAL` while SQLite uses `INTEGER PRIMARY KEY`
- **String concatenation**: PostgreSQL uses `||` (like SQLite) but has additional functions
- **Date functions**: Different function names and formats
- **LIMIT clause**: Same syntax in both databases

### 3. Schema Querying
- **SQLite**: Uses `PRAGMA table_info(table_name)` to get column information
- **PostgreSQL**: Queries the `information_schema` tables to get column information

## Database Schema Details

### Customers Table
```sql
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    city VARCHAR(50),
    state VARCHAR(50),
    email VARCHAR(100),
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Products Table
```sql
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    product_type VARCHAR(50),
    premium_range_min DECIMAL(10, 2),
    premium_range_max DECIMAL(10, 2),
    coverage_limit DECIMAL(12, 2),
    deductible DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Policies Table
```sql
CREATE TABLE policies (
    policy_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    product_id INTEGER REFERENCES products(product_id),
    policy_number VARCHAR(50) UNIQUE NOT NULL,
    start_date DATE,
    end_date DATE,
    premium_amount DECIMAL(10, 2),
    policy_status VARCHAR(20) DEFAULT 'Active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Claims Table
```sql
CREATE TABLE claims (
    claim_id SERIAL PRIMARY KEY,
    policy_id INTEGER REFERENCES policies(policy_id),
    claim_date DATE,
    claim_amount DECIMAL(10, 2),
    settlement_amount DECIMAL(10, 2),
    claim_status VARCHAR(20) DEFAULT 'Pending',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Environment Variables

The application requires the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key for accessing GPT-4o
- `POSTGRES_HOST`: PostgreSQL server host (default: localhost)
- `POSTGRES_PORT`: PostgreSQL server port (default: 5432)
- `POSTGRES_DB`: Database name
- `POSTGRES_USER`: Database username
- `POSTGRES_PASSWORD`: Database password

## Security Considerations

1. **SQL Injection Prevention**: The application relies on the LLM to generate valid SQL, but additional validation could be implemented
2. **API Key Security**: Store API keys in environment variables, not in code
3. **Database Credentials**: Use dedicated database users with minimal required permissions

## Performance Optimization

1. **Connection Pooling**: For production use, consider implementing connection pooling
2. **Query Caching**: Cache frequent queries to reduce database load
3. **Indexing**: Properly index database tables to improve query performance

## Troubleshooting

### Common Issues

1. **Connection Refused**: Verify PostgreSQL server is running and accessible
2. **Authentication Failed**: Check database credentials in environment variables
3. **Table Does Not Exist**: Run `init_database.py` to create tables and populate sample data
4. **API Key Error**: Verify OpenAI API key is correctly set

### Debugging Tips

1. Check the database connection independently before running the app
2. Test SQL queries directly in a PostgreSQL client
3. Monitor the application logs for detailed error messages