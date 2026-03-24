import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    conn = mysql.connector.connect(
        host=os.getenv('MYSQL_HOST', 'localhost'),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', ''),
        database=os.getenv('MYSQL_DATABASE', 'autoforge')
    )
    return conn


def init_db():
    try:
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', '')
        )
        cursor = conn.cursor()

        db_name = os.getenv('MYSQL_DATABASE', 'autoforge')
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        cursor.execute(f"USE {db_name}")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255),
                target_column VARCHAR(255),
                task_type VARCHAR(50),
                best_model VARCHAR(100),
                accuracy FLOAT,
                total_rows INT,
                total_features INT,
                model_path VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("database ready!")

    except Exception as e:
        print(f"db error: {e}")
        print("make sure mysql is running!")


def save_run(filename, target_col, task_type, best_model, accuracy, rows, features, model_path):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO training_runs (filename, target_column, task_type, best_model, accuracy, total_rows, total_features, model_path) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (filename, target_col, task_type, best_model, accuracy, rows, features, model_path)
        )
        conn.commit()
        run_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return run_id
    except Exception as e:
        print(f"error saving run: {e}")
        return None


def get_history():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM training_runs ORDER BY created_at DESC LIMIT 50")
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        for row in results:
            if row.get('created_at'):
                row['created_at'] = str(row['created_at'])
        return results
    except Exception as e:
        print(f"error getting history: {e}")
        return []
