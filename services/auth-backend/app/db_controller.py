"""
NovaCare - Main Integration Controller for Database
Secure connection manager resisting SQL Injections using parameterized queries.
"""
import os
import mysql.connector
from mysql.connector import Error, pooling
import logging

logger = logging.getLogger(__name__)

class DatabaseController:
    """
    Main database connection controller for NovaCare.
    Implements a robust and secure connection pool to the main MySQL engine 
    while keeping the schema abstracted from the calling code.
    All querying methods enforce parameterized query parameters to prevent SQL injection.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseController, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize the connection pool securely using environment variables."""
        self.host = os.getenv("DB_HOST", "192.168.1.15")
        self.port = int(os.getenv("DB_PORT", 3306))
        self.database = os.getenv("DB_NAME", "NovaCare_db")
        
        # Admins will supply their unique username & password via their individual .env files
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        
        self.pool = None
        
        if not self.user or not self.password:
            logger.warning("Database credentials missing in environment variables. DB controller inactive.")
            return

        try:
            # We use a connection pool to manage concurrent database access safely
            self.pool_name = "novacare_pool"
            self.pool_size = 5

            self.pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name=self.pool_name,
                pool_size=self.pool_size,
                pool_reset_session=True,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"Successfully configured DB controller and connected to {self.host}:{self.port} as {self.user}")
        except Error as e:
            logger.error(f"Error while instantiating secure MySQL Connection pool: {e}")
            raise  

    def get_connection(self):
        """Yield robust connections uniformly, drawn securely from the pool."""
        if self.pool is None:
            raise Exception("Database controller rejected connection request: Pool not initialized. Please configure DB_USER and DB_PASSWORD.")
        return self.pool.get_connection()

    def execute_query(self, query: str, params: tuple = None) -> int:
        """
        Executes a data manipulation query (INSERT, UPDATE, DELETE) securely.
        REQUIRED: Passing `params` dynamically uses parameterized payloads to repel typical injection vectors.
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
            return cursor.rowcount
        except Error as e:
            if conn:
                conn.rollback()
            logger.error(f"DB Controller caught query execution error: {e}")
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

    def fetch_all(self, query: str, params: tuple = None) -> list:
        """
        Executes a SELECT query that anticipates retrieving multiple records safely.
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            return cursor.fetchall()
        except Error as e:
            logger.error(f"DB Controller fault mapping dictionary stream securely: {e}")
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

    def fetch_one(self, query: str, params: tuple = None) -> dict:
        """
        Executes a targeted internal lookup or SELECT query expecting an individual entity safely.
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            return cursor.fetchone()
        except Error as e:
            logger.error(f"DB Controller fault resolving lone entity node securely: {e}")
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

# Provide a modular singleton to other elements of the codebase
db = DatabaseController()
