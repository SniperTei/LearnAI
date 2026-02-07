"""
SQLite Database Initialization Script
Creates tables and populates sample data for the Text-to-SQL demo
"""

import sqlite3
import os
from datetime import datetime

def get_db_connection():
    """Establish connection to SQLite database"""
    conn = sqlite3.connect('insurance.db')
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

def create_tables():
    """Create tables in SQLite database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Drop existing tables if they exist
    drop_tables = """
    DROP TABLE IF EXISTS claims;
    DROP TABLE IF EXISTS policies;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS customers;
    """
    
    cursor.executescript(drop_tables)

    # Create customers table
    create_customers = """
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
    """
    
    cursor.execute(create_customers)

    # Create products table
    create_products = """
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
    """
    
    cursor.execute(create_products)

    # Create policies table
    create_policies = """
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
    """
    
    cursor.execute(create_policies)

    # Create claims table
    create_claims = """
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
    """
    
    cursor.execute(create_claims)

    # Commit the changes
    conn.commit()
    conn.close()
    print("Tables created successfully!")

def insert_sample_data():
    """Insert sample data into the tables"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Sample customers
    customers_data = [
        ('John', 'Smith', 35, 'Male', 'New York', 'NY', 'john.smith@email.com', '555-0101', '2023-01-15'),
        ('Emily', 'Johnson', 28, 'Female', 'Los Angeles', 'CA', 'emily.johnson@email.com', '555-0102', '2023-02-20'),
        ('Michael', 'Williams', 45, 'Male', 'Chicago', 'IL', 'michael.williams@email.com', '555-0103', '2023-03-10'),
        ('Sarah', 'Brown', 32, 'Female', 'Houston', 'TX', 'sarah.brown@email.com', '555-0104', '2023-04-05'),
        ('David', 'Jones', 50, 'Male', 'Phoenix', 'AZ', 'david.jones@email.com', '555-0105', '2023-05-12'),
        ('Lisa', 'Garcia', 29, 'Female', 'Philadelphia', 'PA', 'lisa.garcia@email.com', '555-0106', '2023-06-18'),
        ('James', 'Miller', 41, 'Male', 'San Antonio', 'TX', 'james.miller@email.com', '555-0107', '2023-07-22'),
        ('Jennifer', 'Davis', 36, 'Female', 'San Diego', 'CA', 'jennifer.davis@email.com', '555-0108', '2023-08-30'),
        ('Robert', 'Rodriguez', 33, 'Male', 'Dallas', 'TX', 'robert.rodriguez@email.com', '555-0109', '2023-09-14'),
        ('Patricia', 'Martinez', 27, 'Female', 'San Jose', 'CA', 'patricia.martinez@email.com', '555-0110', '2023-10-01')
    ]

    insert_customers = """
    INSERT INTO customers (first_name, last_name, age, gender, city, state, email, phone, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    
    cursor.executemany(insert_customers, customers_data)

    # Sample products
    products_data = [
        ('Basic Health Insurance', 'Health', 150.00, 300.00, 100000.00, 1000.00, '2023-01-01'),
        ('Premium Health Insurance', 'Health', 300.00, 600.00, 500000.00, 500.00, '2023-01-01'),
        ('Comprehensive Auto Insurance', 'Auto', 100.00, 300.00, 100000.00, 500.00, '2023-01-01'),
        ('Basic Auto Insurance', 'Auto', 50, 150.00, 50000.00, 1000.00, '2023-01-01'),
        ('Life Insurance Standard', 'Life', 200.00, 400.00, 250000.00, 0.00, '2023-01-01'),
        ('Life Insurance Premium', 'Life', 400.00, 800.00, 1000000.00, 0.00, '2023-01-01'),
        ('Homeowners Insurance Basic', 'Property', 250.00, 500.00, 200000.00, 1000.00, '2023-01-01'),
        ('Homeowners Insurance Premium', 'Property', 500.00, 1000.00, 500000.00, 500.00, '2023-01-01')
    ]

    insert_products = """
    INSERT INTO products (product_name, product_type, premium_range_min, premium_range_max, coverage_limit, deductible, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    
    cursor.executemany(insert_products, products_data)

    # Sample policies
    policies_data = [
        (1, 1, 'POL001', '2023-02-01', '2024-01-31', 250.00, 'Active', '2023-01-15'),
        (2, 2, 'POL002', '2023-03-01', '2024-02-28', 450.00, 'Active', '2023-02-20'),
        (3, 3, 'POL003', '2023-04-01', '2024-03-31', 200.00, 'Active', '2023-03-10'),
        (4, 4, 'POL004', '2023-05-01', '2024-04-30', 100.00, 'Active', '2023-04-05'),
        (5, 5, 'POL005', '2023-06-01', '2024-05-31', 300.00, 'Active', '2023-05-12'),
        (6, 6, 'POL006', '2023-07-01', '2024-06-30', 600.00, 'Active', '2023-06-18'),
        (7, 7, 'POL007', '2023-08-01', '2024-07-31', 400.00, 'Active', '2023-07-22'),
        (8, 8, 'POL008', '2023-09-01', '2024-08-31', 750.00, 'Active', '2023-08-30'),
        (9, 1, 'POL009', '2023-10-01', '2024-09-30', 250.00, 'Active', '2023-09-14'),
        (10, 2, 'POL010', '2023-11-01', '2024-10-31', 450.00, 'Active', '2023-10-01'),
        (1, 3, 'POL011', '2023-12-01', '2024-11-30', 200.00, 'Pending', '2023-11-20'),
        (2, 4, 'POL012', '2024-01-01', '2024-12-31', 100.00, 'Pending', '2023-12-15')
    ]

    insert_policies = """
    INSERT INTO policies (customer_id, product_id, policy_number, start_date, end_date, premium_amount, policy_status, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """
    
    cursor.executemany(insert_policies, policies_data)

    # Sample claims
    claims_data = [
        (1, '2023-05-15', 5000.00, 4800.00, 'Approved', 'Medical procedure', '2023-05-15'),
        (2, '2023-06-20', 15000.00, 14500.00, 'Approved', 'Emergency room visit', '2023-06-20'),
        (3, '2023-07-10', 8000.00, 7500.00, 'Approved', 'Car accident repair', '2023-07-10'),
        (4, '2023-08-05', 3000.00, 2800.00, 'Approved', 'Minor car damage', '2023-08-05'),
        (5, '2023-09-12', 50000.00, 45000.00, 'Pending', 'Life insurance claim', '2023-09-12'),
        (6, '2023-10-18', 100000.00, 95000.00, 'Approved', 'Life insurance payout', '2023-10-18'),
        (7, '2023-11-22', 25000.00, 24000.00, 'Rejected', 'Home damage - excluded cause', '2023-11-22'),
        (8, '2023-12-30', 75000.00, 70000.00, 'Approved', 'Home fire damage', '2023-12-30')
    ]

    insert_claims = """
    INSERT INTO claims (policy_id, claim_date, claim_amount, settlement_amount, claim_status, description, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    
    cursor.executemany(insert_claims, claims_data)

    # Commit the changes
    conn.commit()
    conn.close()
    print("Sample data inserted successfully!")

if __name__ == "__main__":
    print("Initializing SQLite database...")
    create_tables()
    insert_sample_data()
    print("Database initialization completed!")