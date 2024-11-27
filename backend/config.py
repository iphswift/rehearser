import os

class Config:
    DATABASE = os.path.abspath(os.environ.get('DATABASE', 'papers.db'))
    UPLOAD_FOLDER = os.path.abspath(os.environ.get('UPLOAD_FOLDER', 'uploads'))
    PROCESSED_FOLDER = os.path.abspath(os.environ.get('PROCESSED_FOLDER', 'processed'))
    REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')  # Use as a string
    REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))  # Convert to integer