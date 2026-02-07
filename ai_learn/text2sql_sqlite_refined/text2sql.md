# SQLite Text-to-SQL Implementation Guide

This document explains the SQLite implementation of the Text-to-SQL application, highlighting key aspects of the implementation.

## Database Schema Details

### Customers Table
```sql
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    city TEXT,
    state TEXT,
    email TEXT,
    phone TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Products Table
```sql
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    product_type TEXT,
    premium_range_min REAL,
    premium_range_max REAL,
    coverage_limit REAL,
    deductible REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Policies Table
```sql
CREATE TABLE policies (
    policy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    product_id INTEGER,
    policy_number TEXT UNIQUE NOT NULL,
    start_date DATE,
    end_date DATE,
    premium_amount REAL,
    policy_status TEXT DEFAULT 'Active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);
```

### Claims Table
```sql
CREATE TABLE claims (
    claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id INTEGER,
    claim_date DATE,
    claim_amount REAL,
    settlement_amount REAL,
    claim_status TEXT DEFAULT 'Pending',
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (policy_id) REFERENCES policies (policy_id)
);
```

## Environment Variables

The application requires the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key for accessing GPT-4o

## Security Considerations

1. **SQL Injection Prevention**: The application relies on the LLM to generate valid SQL, but additional validation could be implemented
2. **API Key Security**: Store API keys in environment variables, not in code

## Performance Optimization

1. **Connection Management**: Uses Streamlit's caching for efficient database connections
2. **Query Optimization**: Proper indexing would improve performance on larger datasets

## Troubleshooting

### Common Issues

1. **Database Connection**: Verify that insurance.db exists in the project directory
2. **API Key Error**: Verify OpenAI API key is correctly set
3. **Missing Dependencies**: Run `pip install -r requirements.txt` to install all dependencies

### Debugging Tips

1. Check that the database file exists and is readable
2. Test SQL queries directly using a SQLite client
3. Monitor the application logs for detailed error messages