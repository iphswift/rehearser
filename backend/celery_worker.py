from celery import Celery
from celery.schedules import crontab

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['result_backend'],
        broker=app.config['broker_url']
    )
    celery.conf.update(app.config)

    # Import task modules to ensure they're registered
    import backend.tasks

    # Define periodic tasks
    celery.conf.beat_schedule = {
        "check_stuck_jobs_every_5_minutes": {
            "task": "app.check_stuck_jobs",
            "schedule": crontab(minute="*/1"),  # Run every 5 minutes
        },
    }

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
