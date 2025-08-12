# RAG Ahadees - Hadith Vector Database System

A robust Retrieval-Augmented Generation (RAG) system for Islamic hadiths using **PostgreSQL with pgvector** and Google Gemini embeddings.

## Features

- **Modern Vector Database**: Uses PostgreSQL with pgvector extension for reliable vector storage
- **Environment-Based Configuration**: Secure configuration using environment variables
- **Robust Error Handling**: Automatic retry mechanism with exponential backoff
- **Progress Tracking**: Resume functionality if the process is interrupted
- **Batch Processing**: Efficient batch insertion with configurable batch sizes
- **Connection Testing**: Built-in connection testing before full operation
- **Search Interface**: Interactive search functionality for querying hadiths

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The system now uses `langchain-postgres` with pgvector for better reliability and performance.

### 2. PostgreSQL Setup

#### Prerequisites
- PostgreSQL 11+ with pgvector extension installed
- Python 3.8+

#### Install pgvector Extension
```sql
-- Connect to your PostgreSQL database
CREATE EXTENSION IF NOT EXISTS vector;
```

#### Create Database and User
```sql
CREATE DATABASE langchain;
CREATE USER langchain WITH PASSWORD 'langchain';
GRANT ALL PRIVILEGES ON DATABASE langchain TO langchain;
```

### 3. Configuration Setup

#### Option A: Interactive Setup (Recommended)
Run the interactive setup script:

```bash
python setup_config.py
```

This will guide you through creating your `config.env` file with all necessary settings.

#### Option B: Manual Configuration
Create a `config.env` file in your project directory:

```bash
# PostgreSQL Configuration
POSTGRES_URL=postgresql+psycopg://username:password@localhost:5432/database_name

# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Collection Configuration
COLLECTION_NAME=rag_ahadees

# File Configuration
CSV_FILE_PATH=all_hadiths_clean.csv

# Performance Configuration
BATCH_SIZE=25
USE_BATCH_INSERT=true
MAX_RETRIES=3
RESUME=true
```

#### Option C: System Environment Variables
Set environment variables directly in your system:

```bash
# Windows (PowerShell)
$env:POSTGRES_URL="postgresql+psycopg://username:password@localhost:5432/database_name"
$env:GOOGLE_API_KEY="your_google_api_key"

# Linux/Mac
export POSTGRES_URL="postgresql+psycopg://username:password@localhost:5432/database_name"
export GOOGLE_API_KEY="your_google_api_key"
```

### 4. Test Connection

Before running the full insertion, test your connection:

```bash
python test_postgres_connection.py
```

This will verify:
- PostgreSQL connection and authentication
- pgvector extension functionality
- CSV file readability
- Basic vector operations

## Usage

### 1. Insert Hadiths into Vector Database

```bash
python rag.py
```

### 2. Search Hadiths

```bash
python search_hadiths.py
```

The search script provides:
- **Semantic search** using vector similarity
- **Source-specific filtering** (e.g., search only in Bukhari)
- **Interactive search mode** for real-time queries
- **Score-based ranking** of results

### Configuration Options

All configuration is now handled through environment variables. The main settings are:

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_URL` | PostgreSQL connection string | **Required** |
| `GOOGLE_API_KEY` | Google Gemini API key | **Required** |
| `COLLECTION_NAME` | pgvector collection name | `rag_ahadees` |
| `CSV_FILE_PATH` | Path to CSV file | `all_hadiths_clean.csv` |
| `BATCH_SIZE` | Documents per batch | `25` |
| `USE_BATCH_INSERT` | Enable batch insertion | `true` |
| `MAX_RETRIES` | Max retry attempts | `3` |
| `RESUME` | Enable resume functionality | `true` |

### Key Improvements Made

1. **PostgreSQL pgvector**: More reliable and scalable than cloud-based solutions
2. **Environment Variables**: Secure configuration management
3. **Reduced Batch Size**: From 50 to 25 documents per batch for better stability
4. **Retry Mechanism**: Automatic retries with exponential backoff for failed operations
5. **Progress Tracking**: Saves progress every 10 batches for resume capability
6. **Connection Testing**: Verifies connection before starting insertion
7. **Error Recovery**: Continues processing even if some batches fail
8. **Search Functionality**: Built-in search interface for querying hadiths

### Resume Functionality

If the script is interrupted, it will automatically resume from where it left off when you run it again. Progress is saved to `insertion_progress.json` every 10 successful batches.

## Search Examples

### Basic Search
```python
# Search for hadiths about prayer
results = searcher.search_hadiths("prayer and worship", k=5)
```

### Source-Specific Search
```python
# Search for hadiths about charity in Bukhari
results = searcher.search_by_source("charity and giving", "Bukhari", k=3)
```

### Interactive Search
Run `python search_hadiths.py` for an interactive search interface where you can:
- Enter custom queries
- Specify number of results
- Filter by source
- View detailed metadata

## Security

### API Key Protection
- **Never commit** your `config.env` file to version control
- Use `.gitignore` to exclude configuration files
- Consider using system environment variables for production deployments
- Rotate API keys regularly

### Database Security
- Use strong passwords for database users
- Restrict database access to necessary IP addresses
- Regularly update PostgreSQL and pgvector
- Monitor database access logs

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**: 
   - Run `python setup_config.py` to create configuration
   - Check that `config.env` exists and contains required variables
   - Verify environment variables are properly set

2. **PostgreSQL Connection Issues**:
   - Verify PostgreSQL server is running
   - Check connection string format
   - Ensure pgvector extension is installed
   - Verify user permissions

3. **pgvector Extension Issues**:
   - Ensure PostgreSQL version is 11+
   - Install pgvector: `CREATE EXTENSION vector;`
   - Check extension installation: `\dx vector`

4. **Memory Issues**:
   - Set `USE_BATCH_INSERT=false` for individual document insertion
   - Reduce `BATCH_SIZE`

5. **Performance Issues**:
   - Ensure adequate RAM for PostgreSQL
   - Consider increasing `shared_buffers` in postgresql.conf
   - Monitor query performance with `EXPLAIN ANALYZE`

### Performance Tuning

- **Batch Size**: Start with 25, adjust based on your system and database
- **Retries**: Increase `MAX_RETRIES` for less stable connections
- **Resume**: Enable `RESUME=true` for long-running processes
- **Database**: Optimize PostgreSQL settings for your workload

## File Structure

```
rag-ahadees/
├── rag.py                      # Main RAG insertion script (using PostgreSQL pgvector)
├── search_hadiths.py           # Search interface for querying hadiths
├── test_postgres_connection.py # PostgreSQL connection testing script
├── setup_config.py             # Interactive configuration setup
├── config.env.example          # Configuration template
├── requirements.txt            # Python dependencies (updated for PostgreSQL)
├── all_hadiths_clean.csv      # Hadith data (your CSV file)
└── README.md                  # This file
```

## Data Format

Your CSV should have these columns:
- `id`: Unique identifier
- `hadith_id`: Hadith identifier
- `source`: Source of the hadith
- `chapter_no`: Chapter number
- `hadith_no`: Hadith number
- `chapter`: Chapter name
- `chain_indx`: Chain index
- `text_ar`: Arabic text
- `text_en`: English text

## Monitoring

The script provides detailed logging:
- Progress bars for batch operations
- Success/failure counts
- Detailed error messages
- Progress saving for resume capability
- Configuration validation and masking of sensitive values

## Advanced Features

### Metadata Filtering
The system supports metadata filtering for more precise searches:

```python
# Filter by source and chapter
filter_metadata = {
    "source": "Bukhari",
    "chapter": "Prayer"
}
results = searcher.search_hadiths("morning prayer", k=5, filter_metadata=filter_metadata)
```

### Score-Based Ranking
All search results include similarity scores for better result ranking:

```python
results = searcher.search_hadiths("charity", k=5)
for doc, score in results:
    print(f"Score: {score:.4f} - {doc.page_content[:100]}...")
```

## Support

If you encounter issues:
1. Run `python setup_config.py` to verify configuration
2. Run `test_postgres_connection.py` to verify basic functionality
3. Check the logs for specific error messages
4. Verify environment variables are properly set
5. Ensure your CSV file is properly formatted
6. Verify PostgreSQL and pgvector installation
7. Check database connection and permissions

## License

This project is open source. Please ensure you comply with the terms of use for Google Gemini API and your PostgreSQL instance.
