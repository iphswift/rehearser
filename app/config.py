import os

class Config:
    DATABASE = os.path.abspath(os.environ.get('DATABASE', 'papers.db'))
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')
    UPLOAD_FOLDER = os.path.abspath(os.environ.get('UPLOAD_FOLDER', 'uploads'))
    PROCESSED_FOLDER = os.path.abspath(os.environ.get('PROCESSED_FOLDER', 'processed'))