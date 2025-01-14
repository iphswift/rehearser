import os
import logging

#enable logging
logging.basicConfig(level=logging.DEBUG)

from flask import Flask, request, jsonify, make_response, send_from_directory
from flask_cors import CORS
import sqlite3
from werkzeug.utils import secure_filename
import traceback
from backend.db import save_paper_info, is_file_already_processing_or_processed, init_db
from backend.celery_worker import make_celery
from backend.config import Config
from backend.tasks import process_file

app = Flask(__name__)
CORS(app)

#Load config from app.config
app.config.from_object(Config)
app.config['broker_url'] = 'redis://redis:6379/0'  #celery demands lowercase
app.config['result_backend'] = 'redis://redis:6379/0'

# Ensure the necessary folders exis[pt
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

celery = make_celery(app)

@app.route('/papers', methods=['POST'])
def upload_paper():
    """Upload a new PDF paper and mark it as processing."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    filename = secure_filename(file.filename)

    # Check if the file is already processing or completed
    if is_file_already_processing_or_processed(filename):
        return jsonify({'status': 'error', 'message': 'File is already submitted and processing or completed'}), 400

    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Insert file metadata into the database with 'processing' status
    paper_id = save_paper_info(filename, 'processing')

    # Enqueue the processing task
    process_file.delay(file_path, paper_id)

    # Return an immediate response with the paper ID
    return jsonify({'status': 'success', 'data': {'paper_id': paper_id}}), 201

@app.route('/papers', methods=['GET'])
def list_papers():
    """Retrieve a list of all uploaded papers with their metadata, including status."""
    with sqlite3.connect(app.config['DATABASE']) as conn:
        cursor = conn.cursor()
        # Fetch id, filename, created_at, and status from the papers table
        cursor.execute('SELECT id, filename, created_at, status FROM papers')
        papers = cursor.fetchall()
    
    # Create the response by including the status of each paper
    papers_list = []
    for paper in papers:
        papers_list.append({
            'id': paper[0],
            'filename': paper[1],
            'created_at': paper[2],
            'status': paper[3]  # Include status in the response
        })

    return jsonify({'status': 'success', 'data': papers_list}), 200

@app.route('/papers/<int:paper_id>/status', methods=['GET'])
def get_paper_status(paper_id):
    """Retrieve the status of a specific paper."""
    with sqlite3.connect(app.config['DATABASE']) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT status FROM papers WHERE id = ?', (paper_id,))
        paper = cursor.fetchone()
    
    if paper:
        return jsonify({'status': 'success', 'data': {'status': paper[0]}}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Paper not found'}), 404

@app.route('/papers/<int:paper_id>/audio', methods=['GET'])
def get_paper_audio(paper_id):
    """Retrieve the audio file and speech marks for a specific paper."""
    with sqlite3.connect(app.config['DATABASE']) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT audio_file, speech_marks_file 
            FROM audio_files 
            WHERE paper_id = ?
        ''', (paper_id,))
        audio_info = cursor.fetchone()

    if audio_info:
        return jsonify({
            "audio_file": f"/processed/{os.path.basename(audio_info[0])}",
            "speech_marks_file": f"/processed/{os.path.basename(audio_info[1])}"
        }), 200
    else:
        return jsonify({"error": "Audio files not found for the requested paper."}), 404

@app.route('/processed/<filename>')
def serve_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

@app.errorhandler(500)
def internal_error(error):
    print("An error occurred: ", error)
    print(traceback.format_exc())
    return jsonify({"error": "Internal Server Error"}), 500

@app.route('/papers/<int:paper_id>/narrational_text', methods=['GET'])
def get_paper_narrational_text_url(paper_id):
    """Return the URL of the narrational text for a specific paper"""
    with sqlite3.connect(app.config['DATABASE']) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT filename FROM papers WHERE id = ?', (paper_id,))
        paper = cursor.fetchone()

    if paper:
        narrational_text_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{os.path.splitext(paper[0])[0]}_narrational_text.txt")
        if os.path.exists(narrational_text_path):
            # Return the URL instead of directly downloading the file
            return jsonify({
                'status': 'success',
                'data': {
                    'narrational_text_file': f'/processed/{os.path.basename(narrational_text_path)}'
                }
            }), 200
        else:
            return jsonify({'status': 'error', 'message': 'Narrational text file not found'}), 404
    else:
        return jsonify({'status': 'error', 'message': 'Paper not found'}), 404

@app.route('/papers/<int:paper_id>/pdf', methods=['GET'])
def get_paper_pdf(paper_id):
    """Expose the uploaded PDF file for download by its paper ID."""
    # Retrieve paper metadata from the database
    with sqlite3.connect(app.config['DATABASE']) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT filename FROM papers WHERE id = ?', (paper_id,))
        paper = cursor.fetchone()

    if paper:
        filename = paper[0]
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'File not found on server'}), 404

        with open(file_path, 'rb') as f:
            pdf_data = f.read()

        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'inline; filename="{}"'.format(filename)
        return response


    else:
        return jsonify({'status': 'error', 'message': 'Paper not found'}), 404


@app.route('/papers/<int:paper_id>', methods=['DELETE'])
def delete_paper(paper_id):
    """Delete a paper and all associated files from the database and filesystem."""
    try:
        with sqlite3.connect(app.config['DATABASE']) as conn:
            cursor = conn.cursor()
            
            # Check if the paper exists
            cursor.execute('SELECT filename FROM papers WHERE id = ?', (paper_id,))
            paper = cursor.fetchone()

            if not paper:
                return jsonify({'status': 'error', 'message': 'Paper not found'}), 404
            
            filename = paper[0]
            
            # Remove the paper record from the database
            cursor.execute('DELETE FROM papers WHERE id = ?', (paper_id,))
            cursor.execute('DELETE FROM audio_files WHERE paper_id = ?', (paper_id,))
            conn.commit()

            # Construct file paths for associated files
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            xml_file_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{os.path.splitext(filename)[0]}.xml")
            narrational_text_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{os.path.splitext(filename)[0]}_narrational_text.txt")
            audio_file_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{os.path.splitext(filename)[0]}_combined.wav")
            alignment_file_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{os.path.splitext(filename)[0]}_combined_alignment.json")

            # Delete files if they exist
            for path in [file_path, xml_file_path, narrational_text_path, audio_file_path, alignment_file_path]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Deleted: {path}")
                else:
                    print(f"File not found: {path}")

        return jsonify({'status': 'success', 'message': f'Paper {paper_id} and all associated files have been deleted'}), 200

    except Exception as e:
        print(f"Error deleting paper: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error deleting paper: {str(e)}'}), 500


#####Initialization and Shutdown#####

@app.before_first_request
def initialize():
    init_db()