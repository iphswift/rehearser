from celery import shared_task
from backend.db import update_paper_status, get_stuck_jobs
from backend.utils import is_any_task_running 
from flask import current_app

@shared_task(name='app.check_stuck_jobs')
def check_stuck_jobs():
    """
    Periodically checks for stuck "processing" jobs and marks them as "error"
    if the Celery queue is empty.
    """
    if not is_any_task_running():
        stuck_jobs = get_stuck_jobs()

        for job in stuck_jobs:
            paper_id = job[0]
            print(f"Marking stuck processing job (Paper ID: {paper_id}) as error.")
            update_paper_status(paper_id, 'error')
            
@shared_task(name='app.process_file')
def process_file(file_path, paper_id):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = current_app.config['PROCESSED_FOLDER']
    xml_output_path = os.path.join(output_dir, f"{base_name}.xml")

    try:
        # Extract narrational text from PDF
        narrational_text = get_narrational_text(file_path, base_name, output_dir, xml_output_path)

        # Use FastSpeech2 to generate speech and perform alignment
        audio_file, combined_alignment_file = text_to_speech_fastspeech(narrational_text, output_dir, base_name)

        # Update the status to 'completed'
        update_paper_status(paper_id, 'completed')

        # Save audio and alignment information to the database
        save_audio_info(paper_id, audio_file, combined_alignment_file)

    except Exception as e:
        # Update the status to 'error'
        print(f"Processing failed for paper {paper_id}: {str(e)}")
        traceback.print_exc()
        update_paper_status(paper_id, 'error')