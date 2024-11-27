import redis
from flask import current_app

def is_any_task_running():
    redis_client = redis.StrictRedis(
        host=current_app.config['REDIS_HOST'],
        port=current_app.config['REDIS_PORT'],
        db=0
    )
    active_queue_keys = redis_client.keys("celery@*.active")
    reserved_queue_keys = redis_client.keys("celery@*.reserved")
    scheduled_queue_keys = redis_client.keys("celery@*.scheduled")
    
    for queue_key in active_queue_keys + reserved_queue_keys + scheduled_queue_keys:
        if redis_client.llen(queue_key) > 0:
            return True
    
    return False