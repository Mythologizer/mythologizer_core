#!/usr/bin/env python3

import logging
import os
from dotenv import load_dotenv, find_dotenv
import psycopg
from mythologizer_postgres.db import ping_db_basic, check_if_tables_exist

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection using environment variables."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "")
    password = os.getenv("POSTGRES_PASSWORD", "")
    dbname = os.getenv("POSTGRES_DB", "")
    
    return psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )

def check_vector_extension():
    """Check the vector extension status."""
    try:
        # First check basic connection
        logger.info("Testing basic database connection...")
        connection_status = ping_db_basic()
        logger.info(f"Database connection: {connection_status}")
        
        if not connection_status:
            logger.error("Cannot connect to database")
            return
        
        # Check if tables exist
        logger.info("Checking if tables exist...")
        expected_tables = ["mythemes", "myths", "agent_attribute_defs"]
        table_status = check_if_tables_exist(expected_tables)
        logger.info(f"Table status: {table_status}")
        
        # Connect directly to check vector extension
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if vector extension is installed
                cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
                vector_ext = cur.fetchone()
                logger.info(f"Vector extension: {vector_ext}")
                
                if vector_ext:
                    # Check vector types
                    cur.execute("SELECT typname, typlen, typalign FROM pg_type WHERE typname LIKE 'vector%'")
                    vector_types = cur.fetchall()
                    logger.info(f"Vector types: {vector_types}")
                    
                    # Check if mythemes table exists and its structure
                    cur.execute("""
                        SELECT column_name, data_type, udt_name, character_maximum_length
                        FROM information_schema.columns 
                        WHERE table_name = 'mythemes'
                        ORDER BY ordinal_position
                    """)
                    columns = cur.fetchall()
                    logger.info(f"Mythemes table columns: {columns}")
                    
                    # Try to create a simple vector
                    try:
                        cur.execute("SELECT '[1,2,3]'::vector(3)")
                        test_vector = cur.fetchone()
                        logger.info(f"Test vector creation: {test_vector}")
                    except Exception as e:
                        logger.error(f"Failed to create test vector: {e}")
                        
                else:
                    logger.error("Vector extension not found!")
                    
    except Exception as e:
        logger.error(f"Database check failed: {e}")

def test_vector_insert():
    """Test inserting a vector into the mythemes table."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Try different vector insertion methods
                
                # Method 1: Raw list
                logger.info("Testing vector insert with raw list...")
                cur.execute("""
                    INSERT INTO public.mythemes (sentence, embedding)
                    VALUES (%s, %s)
                """, ("test1", [1.0, 2.0, 3.0, 4.0, 5.0]))
                
                # Method 2: Vector string without type cast
                logger.info("Testing vector insert with vector string...")
                cur.execute("""
                    INSERT INTO public.mythemes (sentence, embedding)
                    VALUES (%s, %s)
                """, ("test2", "[1,2,3,4,5]"))
                
                # Method 3: Using vector() function
                logger.info("Testing vector insert with vector() function...")
                cur.execute("""
                    INSERT INTO public.mythemes (sentence, embedding)
                    VALUES (%s, vector(%s))
                """, ("test3", "[1,2,3,4,5]"))
                
                logger.info("All test vector inserts successful")
                
                # Clean up
                cur.execute("DELETE FROM public.mythemes WHERE sentence LIKE 'test%'")
                conn.commit()
                
    except Exception as e:
        logger.error(f"Test vector insert failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    logger.info("Checking vector extension and database schema...")
    check_vector_extension()
    
    logger.info("Testing vector insert...")
    test_vector_insert()
