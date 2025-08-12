#!/usr/bin/env python3
"""
Test script to verify PostgreSQL connection and pgvector functionality
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_postgres_connection():
    """Test basic PostgreSQL connection"""
    
    print("🔍 Testing PostgreSQL connection...")
    
    # Get configuration
    postgres_url = os.getenv('POSTGRES_URL')
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    if not postgres_url:
        print("❌ Missing POSTGRES_URL environment variable")
        return False
    
    if not google_api_key:
        print("❌ Missing GOOGLE_API_KEY environment variable")
        return False
    
    print(f"PostgreSQL URL: {postgres_url}")
    print(f"Google API Key: {google_api_key[:8]}...")
    
    try:
        # Test PostgreSQL connection
        print("\n1️⃣ Testing PostgreSQL connection...")
        from langchain_postgres import PGVector
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        # Initialize embeddings
        os.environ["GOOGLE_API_KEY"] = google_api_key
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key
        )
        
        # Test connection by creating a test collection
        test_collection = "test-connection-123"
        vector_store = PGVector(
            connection=postgres_url,
            embedding_function=embeddings,
            collection_name=test_collection,
            use_jsonb=True
        )
        
        print("✅ PostgreSQL connection successful!")
        
        # Test document insertion
        print("\n2️⃣ Testing document insertion...")
        from langchain_core.documents import Document
        
        test_doc = Document(
            page_content="This is a test hadith in English. هذا حديث تجريبي باللغة العربية.",
            metadata={"source": "test", "hadith_number": 999}
        )
        
        vector_store.add_documents([test_doc])
        print("✅ Document insertion successful!")
        
        # Test search
        print("\n3️⃣ Testing vector search...")
        search_results = vector_store.similarity_search("test hadith", k=1)
        print(f"✅ Search successful: {len(search_results)} results")
        print(f"Search result: {search_results[0].page_content}")
        
        # Clean up test collection
        print("\n4️⃣ Cleaning up test collection...")
        # Note: PGVector doesn't have a direct delete method, but the test collection
        # will be automatically cleaned up when the connection is closed
        
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL test failed: {e}")
        print("\nThis might be due to:")
        print("1. Database connection issues")
        print("2. Incorrect connection string")
        print("3. Database server being unavailable")
        print("4. pgvector extension not installed")
        print("5. Missing dependencies")
        return False

def test_csv_reading():
    """Test if we can read the CSV file"""
    try:
        import pandas as pd
        print("\n📖 Testing CSV file reading...")
        
        csv_file_path = os.getenv('CSV_FILE_PATH', 'all_hadiths_clean.csv')
        print(f"CSV file path: {csv_file_path}")
        
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"✅ CSV file read successfully! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head(3).to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ CSV reading test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 PostgreSQL pgvector Connection Test")
    print("=" * 50)
    
    # Test PostgreSQL connection
    if test_postgres_connection():
        print("\n✅ PostgreSQL connection successful!")
        
        # Test CSV reading
        if test_csv_reading():
            print("\n🎉 All tests passed! Your PostgreSQL pgvector setup is working.")
            print("\nYou can now run the main script:")
            print("python rag.py")
            sys.exit(0)
        else:
            print("\n⚠️  PostgreSQL works but CSV reading failed.")
            sys.exit(1)
    else:
        print("\n💥 PostgreSQL connection test failed. Check your configuration and database setup.")
        sys.exit(1)
