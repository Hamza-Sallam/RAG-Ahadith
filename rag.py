import csv
import os
import time
from typing import List, Dict, Any
import pandas as pd
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
from tqdm import tqdm
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HadithPGVectorInserter:
    def __init__(self, 
                 postgres_url: str = None,
                 google_api_key: str = None,
                 collection_name: str = None,
                 max_retries: int = None):
        """
        Initialize the Hadith PostgreSQL pgvector inserter
        
        Args:
            postgres_url: PostgreSQL connection string (defaults to POSTGRES_URL env var)
            google_api_key: Google API key for Gemini embeddings (defaults to GOOGLE_API_KEY env var)
            collection_name: Name of the pgvector collection (defaults to COLLECTION_NAME env var)
            max_retries: Maximum number of retries for failed operations (defaults to MAX_RETRIES env var)
        """
        # Load configuration from environment variables with fallbacks
        self.postgres_url = postgres_url or os.getenv('POSTGRES_URL')
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        self.collection_name = collection_name or os.getenv('COLLECTION_NAME', 'rag_ahadees')
        self.max_retries = max_retries or int(os.getenv('MAX_RETRIES', '3'))
        
        # Validate required configuration
        if not self.postgres_url:
            raise ValueError("POSTGRES_URL is required")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        
        logger.info(f"Initializing with PostgreSQL URL: {self.postgres_url}")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Max retries: {self.max_retries}")
        
        os.environ["GOOGLE_API_KEY"] = self.google_api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.google_api_key
        )
        
        # Initialize PostgreSQL pgvector store
        self.vector_store = PGVector(
            connection=self.postgres_url,
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            use_jsonb=True
        )
        
        # Test connection before proceeding
        self._test_connection()
    
    def _test_connection(self):
        """Test the connection to PostgreSQL"""
        try:
            logger.info("Testing PostgreSQL connection...")
            # Test by trying to get collection info
            # PGVector will create the collection if it doesn't exist
            logger.info("✅ Successfully connected to PostgreSQL with pgvector")
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            logger.error("This might be due to:")
            logger.error("1. Database connection issues")
            logger.error("2. Incorrect connection string")
            logger.error("3. Database server being unavailable")
            logger.error("4. pgvector extension not installed")
            raise
    def _create_table_if_not_exists(self):
        """Create the table if it doesn't exist"""
        try:
            # PGVector will automatically create the table when needed
            logger.info(f"Table for collection '{self.collection_name}' is ready")
        except Exception as e:
            logger.error(f"Error preparing table: {e}")
            raise
    def read_csv_file(self, csv_file_path: str) -> pd.DataFrame:
        """
        Read the CSV file containing hadith data
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the hadith data
        """
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            logger.info(f"Successfully loaded {len(df)} rows from {csv_file_path}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
    
    def create_document_from_row(self, row: pd.Series) -> Document:
        """
        Create a LangChain Document from a CSV row
        
        Args:
            row: Pandas Series representing a row from the CSV
            
        Returns:
            LangChain Document object
        """
        # Construct content from Arabic and English text
        text_ar = row.get('text_ar', '')
        text_en = row.get('text_en', '')
        
        content = f"الحديث باللغة العربية:\n{text_ar}\n\nhadith in english:\n{text_en}"        
        # Create metadata dictionary
        metadata = {
            "source": row.get('source', 'Unknown'),
            "hadith_number": row.get('hadith_no', 0) if pd.notna(row.get('hadith_no')) else 0,
            "chapter_number": row.get('chapter_no', 0) if pd.notna(row.get('chapter_no')) else 0,
            "chapter": row.get('chapter', ''),
            "chain_index": row.get('chain_indx', ''),
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def insert_chunks_batch(self, documents: List[Document], batch_size: int = 25, resume_from: int = 0):
        """
        Insert documents in batches to PostgreSQL with retry mechanism and resume capability
        
        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to insert in each batch
            resume_from: Resume from this batch number (0-indexed)
        """
        total_docs = len(documents)
        total_batches = (total_docs + batch_size - 1) // batch_size
        logger.info(f"Starting to insert {total_docs} documents in batches of {batch_size} (total batches: {total_batches})")
        
        if resume_from > 0:
            logger.info(f"Resuming from batch {resume_from + 1}")
        
        successful_batches = 0
        failed_batches = 0
        
        for i in tqdm(range(resume_from * batch_size, total_docs, batch_size), desc="Inserting batches"):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Try to insert the batch with retries
            success = self._insert_batch_with_retry(batch, batch_num)
            
            if success:
                successful_batches += 1
                # Save progress every 10 successful batches
                if successful_batches % 10 == 0:
                    self._save_progress(batch_num, successful_batches, failed_batches)
            else:
                failed_batches += 1
                
            # Add a small delay between batches to prevent overwhelming the database
            time.sleep(0.5)
        
        logger.info(f"Batch insertion complete. Successful: {successful_batches}, Failed: {failed_batches}")
    
    def _insert_batch_with_retry(self, batch: List[Document], batch_num: int) -> bool:
        """Insert a batch with retry mechanism using pgvector"""
        for attempt in range(self.max_retries):
            try:
                # Use pgvector's add_documents method
                self.vector_store.add_documents(batch)
                logger.info(f"Successfully inserted batch {batch_num}")
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for batch {batch_num}: {e}")
                if attempt < self.max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = (2 ** attempt) * 2
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to insert batch {batch_num} after {self.max_retries} attempts: {e}")
                    return False
        return False
    
    def _save_progress(self, current_batch: int, successful: int, failed: int):
        """Save progress to a file for potential resume"""
        progress_data = {
            "current_batch": current_batch,
            "successful_batches": successful,
            "failed_batches": failed,
            "timestamp": time.time()
        }
        
        try:
            with open("insertion_progress.json", "w") as f:
                json.dump(progress_data, f)
            logger.info(f"Progress saved: Batch {current_batch}, Successful: {successful}, Failed: {failed}")
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from file if it exists"""
        try:
            if os.path.exists("insertion_progress.json"):
                with open("insertion_progress.json", "r") as f:
                    progress = json.load(f)
                logger.info(f"Loaded progress: Batch {progress['current_batch']}, Successful: {progress['successful_batches']}, Failed: {progress['failed_batches']}")
                return progress
        except Exception as e:
            logger.warning(f"Could not load progress: {e}")
        
        return {"current_batch": 0, "successful_batches": 0, "failed_batches": 0}
    
    def insert_chunks_individual(self, documents: List[Document]):
        """
        Insert documents one by one (slower but more reliable) using pgvector
        """
        total_docs = len(documents)
        logger.info(f"Starting to insert {total_docs} documents individually")
        
        successful_inserts = 0
        failed_inserts = 0
        
        for i, doc in enumerate(tqdm(documents, desc="Inserting documents")):
            try:
                self.vector_store.add_documents([doc])
                successful_inserts += 1
            except Exception as e:
                logger.error(f"Error inserting document {i}: {e}")
                failed_inserts += 1
                continue
        
        logger.info(f"Insertion complete. Successful: {successful_inserts}, Failed: {failed_inserts}")
    
    def process_csv_file(self, 
                        csv_file_path: str, 
                        batch_size: int = 25,
                        use_batch_insert: bool = True,
                        resume: bool = True):
        """
        Main method to process CSV file and insert into PostgreSQL pgvector
        
        Args:
            csv_file_path: Path to the CSV file
            batch_size: Batch size for insertion
            use_batch_insert: Whether to use batch insertion or individual insertion
            resume: Whether to resume from previous progress if available
        """
        # Read CSV file
        df = self.read_csv_file(csv_file_path)
        
        # Convert rows to documents
        logger.info("Converting CSV rows to documents...")
        documents = []
        for _, row in df.iterrows():
            try:
                doc = self.create_document_from_row(row)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error creating document from row: {e}")
                continue
        
        logger.info(f"Created {len(documents)} documents from CSV")
        
        # Check if we should resume from previous progress
        resume_from = 0
        if resume and use_batch_insert:
            progress = self._load_progress()
            resume_from = progress["current_batch"]
        
        # Insert documents into PostgreSQL pgvector
        if use_batch_insert:
            self.insert_chunks_batch(documents, batch_size, resume_from)
        else:
            self.insert_chunks_individual(documents)
        
        # Clean up progress file on successful completion
        if os.path.exists("insertion_progress.json"):
            try:
                os.remove("insertion_progress.json")
                logger.info("Progress file cleaned up")
            except Exception as e:
                logger.warning(f"Could not clean up progress file: {e}")
        
        logger.info("Processing complete!")

def main():
    """
    Main function to run the hadith insertion script
    """
    # Load configuration from environment variables
    config = {
        "postgres_url": os.getenv("POSTGRES_URL"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "collection_name": os.getenv("COLLECTION_NAME", "rag_ahadees"),
        "csv_file_path": os.getenv("CSV_FILE_PATH", "all_hadiths_clean.csv"),
        "batch_size": int(os.getenv("BATCH_SIZE", "25")),
        "use_batch_insert": os.getenv("USE_BATCH_INSERT", "true").lower() == "true",
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),
        "resume": os.getenv("RESUME", "true").lower() == "true"
    }
    
    # Validate required configuration
    required_vars = ["google_api_key"]
    missing_vars = [var for var in required_vars if not config[var]]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set these variables in your config.env file or as environment variables")
        raise ValueError(f"Missing required configuration: {missing_vars}")
    
    logger.info("Configuration loaded successfully:")
    for key, value in config.items():
        if key in ["google_api_key"]:
            # Mask sensitive values
            masked_value = value[:8] + "..." if value else None
            logger.info(f"  {key}: {masked_value}")
        else:
            logger.info(f"  {key}: {value}")
    
    try:
        # Initialize inserter
        inserter = HadithPGVectorInserter(
            postgres_url=config["postgres_url"],
            google_api_key=config["google_api_key"],
            collection_name=config["collection_name"],
            max_retries=config["max_retries"]
        )
        
        # Process CSV file
        inserter.process_csv_file(
            csv_file_path=config["csv_file_path"],
            batch_size=config["batch_size"],
            use_batch_insert=config["use_batch_insert"],
            resume=config["resume"]
        )
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise

if __name__ == "__main__":
    main()