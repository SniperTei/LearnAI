"""
Test script to verify PostgreSQL connection and basic operations
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', 5432),
            database=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD')
        )
        
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        
        print("✅ PostgreSQL connection successful!")
        print(f"PostgreSQL version: {db_version[0]}")
        
        # Check if our tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name IN ('customers', 'products', 'policies', 'claims');
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        
        if tables:
            print(f"✅ Found tables: {', '.join(tables)}")
        else:
            print("❌ No expected tables found. Run init_database.py to create them.")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("\nMake sure:")
        print("- PostgreSQL server is running")
        print("- Environment variables are set correctly")
        print("- Database user has necessary permissions")
        return False

if __name__ == "__main__":
    print("Testing PostgreSQL connection...")
    test_connection()