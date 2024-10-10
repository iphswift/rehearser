import sqlite3
from datetime import datetime, timedelta
from app.config import Config

def save_paper_info(filename, status):
    """Save the paper metadata to the database with an initial status."""
    created_at = datetime.utcnow().isoformat()
    with sqlite3.connect(Config.DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO papers (filename, created_at, status)
            VALUES (?, ?, ?)
        ''', (filename, created_at, status))
        paper_id = cursor.lastrowid
        conn.commit()
    return paper_id

def save_audio_info(paper_id, audio_file, speech_marks_file):
    with sqlite3.connect(Config.DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audio_files (paper_id, audio_file, speech_marks_file)
            VALUES (?, ?, ?)
        ''', (paper_id, audio_file, speech_marks_file))
        conn.commit()

def is_file_already_processing_or_processed(filename):
    """Check if the file has already been submitted and is either processing or completed."""
    with sqlite3.connect(Config.DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT status FROM papers WHERE filename = ?', (filename,))
        result = cursor.fetchone()
    
    if result and result[0] in ['processing', 'completed']:
        return True  # Return True if the file is either processing or already completed
    return False  # Allow resubmission if the status is 'error'


def update_paper_status(paper_id, status):
    """Update the status of a paper in the database."""
    with sqlite3.connect(Config.DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE papers SET status = ? WHERE id = ?', 
                        (status, paper_id))
        conn.commit()

def mark_unfinished_jobs_as_error():
    """Mark papers that were 'processing' when the app exited as 'error'."""
    with sqlite3.connect(Config.DATABASE) as conn:
        cursor = conn.cursor()
        one_day_ago = datetime.now() - timedelta(hours=6)
        cursor.execute('UPDATE papers SET status = "error" WHERE status = "processing" AND created_at < ?', (one_day_ago.strftime('%Y-%m-%d %H:%M:%S'),))
        conn.commit()

def init_db():
    with sqlite3.connect(Config.DATABASE) as conn:
        cursor = conn.cursor()
        try:
            # Check if the 'status' column exists in the 'papers' table
            cursor.execute("PRAGMA table_info(papers)")
            columns = [column[1] for column in cursor.fetchall()]

            # Create the 'papers' table if it doesn't exist
            cursor.execute('''CREATE TABLE IF NOT EXISTS papers (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            filename TEXT NOT NULL,
                            created_at TEXT NOT NULL)''')

            # Add the 'status' column if it doesn't exist
            if 'status' not in columns:
                cursor.execute('ALTER TABLE papers ADD COLUMN status TEXT DEFAULT "complete"')

            if 'narrational_text' in columns:
                cursor.execute('ALTER TABLE papers DROP COLUMN narrational_text')

            # Create the 'audio_files' table if it doesn't exist
            cursor.execute('''CREATE TABLE IF NOT EXISTS audio_files (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            paper_id INTEGER NOT NULL,
                            audio_file TEXT NOT NULL,
                            speech_marks_file TEXT NOT NULL,
                            FOREIGN KEY (paper_id) REFERENCES papers(id))''')
            print("Created 'papers' table.")
        except sqlite3.Error as e:
            print(f"Error creating 'papers' table: {e}")
        
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    audio_file TEXT NOT NULL,
                    speech_marks_file TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(id)
                )
            ''')
            print("Created 'audio_files' table.")
        except sqlite3.Error as e:
            print(f"Error creating 'audio_files' table: {e}")
        
        conn.commit()