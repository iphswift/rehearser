import os
import re
import torch
import torchaudio
from espnet2.bin.tts_inference import Text2Speech
from espnet_model_zoo.downloader import ModelDownloader
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
import json
from nltk.corpus import cmudict
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Necessary preload for ESPNet g2p conversion
cmu = cmudict.dict()

tts_model = None  # Global variable for the TTS model in each process

def init_tts_model():
    global tts_model
    d = ModelDownloader()
    model_name = "kan-bayashi/ljspeech_fastspeech2"
    model_path = d.download(model_name)
    tts_model = Text2Speech.from_pretrained(model_path)

def split_text(text, max_length=222):
    sentences = re.split(r'(?<=[.!?,;:])\s+', text)
    chunks = []
    current_chunk = ""

    for i, sentence in enumerate(sentences):
        words = sentence.split()

        for word in words:
            if len(current_chunk) + len(word) + 1 <= max_length:
                current_chunk += (" " + word if current_chunk else word)
            else:
                chunks.append(current_chunk)
                current_chunk = word

        if i + 1 < len(sentences) and len(current_chunk) + len(sentences[i + 1]) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def process_chunk(chunk, output_dir, base_name, index):
    """Process individual chunks in parallel."""
    global tts_model
    audio_path = os.path.join(output_dir, f"{base_name}_part_{index}.wav")
    
    logging.info(f"Processing chunk {index}: {chunk[:30]}...")

    try:
        with torch.no_grad():
            wav = tts_model(chunk)["wav"]
            torchaudio.save(audio_path, wav.unsqueeze(0).cpu(), tts_model.fs)
        
        # Calculate duration
        audio_duration = len(wav) / tts_model.fs  # Duration in seconds
        alignment = {
            "duration": audio_duration,
            "text": chunk
        }
        logging.info(f"Chunk {index} processed: audio saved to {audio_path}, duration: {audio_duration}s")
        return audio_path, alignment
    except Exception as e:
        logging.error(f"Error processing chunk {index}: {e}")
        raise

def text_to_speech_fastspeech(text, output_dir, base_name, max_workers=4):
    """Convert text to speech using FastSpeech2 with parallel chunk processing."""
    combined_audio_path = os.path.join(output_dir, f"{base_name}_combined.wav")
    combined_alignment_path = os.path.join(output_dir, f"{base_name}_combined_alignment.json")

    if os.path.exists(combined_audio_path) and os.path.exists(combined_alignment_path):
        logging.info(f"Combined audio and alignment already exist: {combined_audio_path}, {combined_alignment_path}")
        return combined_audio_path, combined_alignment_path

    global tts_model
    if tts_model is None:
        logging.info("Initializing TTS model...")
        init_tts_model()

    logging.info("Splitting text into chunks...")
    chunks = split_text(text)
    logging.info(f"Text split into {len(chunks)} chunks.")

    audio_segments = [None] * len(chunks)  # Initialize with None to store in correct order later
    alignments = [None] * len(chunks)  # Same for alignments
    generated_files = []  # Track generated files to clean up in case of error

    try:
        # Use ThreadPoolExecutor to process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, chunk, output_dir, base_name, i): i for i, chunk in enumerate(chunks)}

            for future in as_completed(futures):
                i = futures[future]  # Get the original index for correct ordering
                try:
                    audio_path, alignment = future.result()
                    audio_segments[i] = audio_path  # Store audio segment at the correct index
                    alignments[i] = alignment  # Store alignment at the correct index
                    generated_files.append(audio_path)  # Track files for potential cleanup
                    logging.info(f"Chunk {i} completed successfully.")
                except Exception as e:
                    logging.error(f"Chunk {i} failed: {e}")
                    raise  # Re-raise the exception to be caught by the outer try block

        # Adjust offsets and combine outputs
        combined_alignments = []
        current_time_offset = 0.0
        for alignment in alignments:
            combined_alignments.append({
                "begin": str(current_time_offset),
                "end": str(current_time_offset + alignment["duration"]),
                "text": alignment["text"]
            })
            current_time_offset += alignment["duration"]

        # Combine all audio segments
        logging.info("Combining audio segments...")
        combined_audio = AudioSegment.empty()
        for audio_file_path in audio_segments:
            audio_segment = AudioSegment.from_wav(audio_file_path)
            combined_audio += audio_segment

        combined_audio.export(combined_audio_path, format="wav")
        logging.info(f"Combined audio saved to {combined_audio_path}.")

        # Write combined alignment to file
        with open(combined_alignment_path, 'w') as f:
            json.dump({"fragments": combined_alignments}, f, indent=4)
        logging.info(f"Combined alignment saved to {combined_alignment_path}.")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        # Clean up partial files
        for file in generated_files:
            if os.path.exists(file):
                os.remove(file)
                logging.info(f"Cleaned up partial file: {file}")
        # Clean up any incomplete combined files
        if os.path.exists(combined_audio_path):
            os.remove(combined_audio_path)
            logging.info(f"Removed incomplete combined audio: {combined_audio_path}")
        if os.path.exists(combined_alignment_path):
            os.remove(combined_alignment_path)
            logging.info(f"Removed incomplete combined alignment: {combined_alignment_path}")
        raise  # Re-raise the error after cleaning up

    finally:
        # Ensure cleanup of partial files even in case of an unexpected failure
        for audio_file_path in audio_segments:
            if audio_file_path and os.path.exists(audio_file_path):
                os.remove(audio_file_path)
                logging.info(f"Deleted partial audio file: {audio_file_path}")

    return combined_audio_path, combined_alignment_path