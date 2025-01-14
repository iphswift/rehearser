import os
import re
import torch
import torchaudio
from espnet2.bin.tts_inference import Text2Speech
from espnet_model_zoo.downloader import ModelDownloader
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from typing import List, Dict, Tuple, Any
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

def process_chunk(chunk: Dict[str, Any], output_dir: str, base_name: str, index: int) -> Tuple[str, Dict[str, Any]]:
    """Process individual chunks in parallel."""
    global tts_model
    audio_path = os.path.join(output_dir, f"{base_name}_part_{index}.wav")
    
    # Validate chunk structure
    if not isinstance(chunk, dict):
        logging.error(f"Chunk {index} is not a dictionary: {type(chunk)}")
        raise TypeError(f"Chunk {index} is not a dictionary.")
    
    required_keys = {'narrational_text_block', 'bounding_box', 'page_number'}  # Updated to 'scratch_text'
    if not required_keys.issubset(chunk.keys()):
        missing_keys = required_keys - chunk.keys()
        logging.error(f"Chunk {index} is missing required keys: {missing_keys}")
        raise KeyError(f"Chunk {index} is missing required keys: {missing_keys}")
    
    # Safely access 'scratch_text' and 'bounding_box' from the chunk dictionary
    text = chunk.get('narrational_text_block', '')
    bounding_box = chunk.get('bounding_box', [0.0, 0.0, 0.0, 0.0])  # Default bbox if not present
    page_number = chunk.get('page_number', None)

    # Ensure 'page_number' is present and is an integer
    if page_number is None:
        logging.error(f"Chunk {index} is missing 'page_number'.")
        raise KeyError(f"Chunk {index} is missing 'page_number'.")
    if not isinstance(page_number, int):
        logging.error(f"Chunk {index} 'page_number' is not an integer: {type(page_number)}")
        raise TypeError(f"Chunk {index} 'page_number' is not an integer.")

    
    # Ensure 'text' is a string
    if not isinstance(text, str):
        logging.error(f"Chunk {index} 'scratch_text' is not a string: {type(text)}")
        raise TypeError(f"Chunk {index} 'scratch_text' is not a string.")
    
    # Log the type and content of 'text' and 'bounding_box'
    logging.debug(f"Chunk {index} type: {type(chunk)}")
    logging.debug(f"Chunk {index} scratch_text type: {type(text)}")
    logging.debug(f"Chunk {index} scratch_text: {text}")
    logging.debug(f"Chunk {index} bounding_box: {bounding_box}")
    
    # Log the first 30 characters of the text
    logging.info(f"Processing chunk {index}: Page {page_number}, Text: {text[:30]}...")
    
    try:
        with torch.no_grad():
            # Generate speech waveform using the TTS model
            wav = tts_model(text)["wav"]
            # Save the waveform to a WAV file
            torchaudio.save(audio_path, wav.unsqueeze(0).cpu(), tts_model.fs)
        
        # Calculate duration of the audio in seconds
        audio_duration = len(wav) / tts_model.fs  # Duration in seconds
        
        # Create the alignment dictionary including 'bounding_box'
        alignment = {
            "duration": audio_duration,
            "text": text,  # Alignment still uses 'text'
            "bounding_box": bounding_box,
            "page_number": page_number  # Include 'page_number'
        }

        
        logging.info(f"Chunk {index} processed: audio saved to {audio_path}, duration: {audio_duration}s")
        return audio_path, alignment
    except Exception as e:
        logging.error(f"Error processing chunk {index}: {e}")
        raise

def text_to_speech_fastspeech(chunks: List[Dict[str, Any]], output_dir: str, base_name: str, max_workers: int = 4) -> Tuple[str, str]:
    """
    Convert text to speech using FastSpeech2 with parallel chunk processing.
    Includes bounding box information in the alignment.

    Args:
        chunks (List[Dict[str, Any]]): List of chunk dictionaries containing 'scratch_text', 'bounding_box'.
        output_dir (str): Directory to save the output audio and alignment files.
        base_name (str): Base name for the output files.
        max_workers (int, optional): Maximum number of parallel workers. Defaults to 4.

    Returns:
        Tuple[str, str]: Paths to the combined audio file and the combined alignment JSON.
    """
    combined_audio_path = os.path.join(output_dir, f"{base_name}_combined.wav")
    combined_alignment_path = os.path.join(output_dir, f"{base_name}_combined_alignment.json")

    # Validate chunks
    if not isinstance(chunks, list):
        logging.error("Chunks should be a list of dictionaries.")
        raise TypeError("Chunks should be a list of dictionaries.")

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            logging.error(f"Chunk {i} is not a dictionary: {type(chunk)}")
            raise TypeError(f"Chunk {i} is not a dictionary.")
        required_keys = {'narrational_text_block', 'bounding_box', 'page_number'}  # Include 'page_number'
        if not required_keys.issubset(chunk.keys()):
            missing_keys = required_keys - chunk.keys()
            logging.error(f"Chunk {i} is missing required keys: {missing_keys}")
            raise KeyError(f"Chunk {i} is missing required keys: {missing_keys}")
        if not isinstance(chunk['narrational_text_block'], str):
            logging.error(f"Chunk {i} 'narrational_text_block' is not a string: {type(chunk['narrational_text_block'])}")
            raise TypeError(f"Chunk {i} 'narrational_text_block' is not a string.")
        if not (isinstance(chunk['bounding_box'], (list, tuple)) and len(chunk['bounding_box']) == 4):
            logging.error(f"Chunk {i} 'bounding_box' must be a list or tuple of four floats.")
            raise ValueError(f"Chunk {i} 'bounding_box' must be a list or tuple of four floats.")
        if not isinstance(chunk['page_number'], int):
            logging.error(f"Chunk {i} 'page_number' is not an integer: {type(chunk['page_number'])}")
            raise TypeError(f"Chunk {i} 'page_number' is not an integer.")


    if os.path.exists(combined_audio_path) and os.path.exists(combined_alignment_path):
        logging.info(f"Combined audio and alignment already exist: {combined_audio_path}, {combined_alignment_path}")
        return combined_audio_path, combined_alignment_path

    global tts_model
    if tts_model is None:
        logging.info("Initializing TTS model...")
        init_tts_model()

    audio_segments = [None] * len(chunks)  # Initialize with None to store in correct order later
    alignments = [None] * len(chunks)  # Same for alignments
    generated_files = []  # Track generated files to clean up in case of error

    try:
        # Use ThreadPoolExecutor to process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit each chunk for processing
            futures = {
                executor.submit(process_chunk, chunk, output_dir, base_name, i): i
                for i, chunk in enumerate(chunks)
            }

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
        for i, alignment in enumerate(alignments):
            if not alignment:
                logging.warning(f"No alignment data for chunk {i}. Skipping.")
                continue

            # Ensure alignment contains 'duration', 'text', and 'bounding_box'
            duration = alignment.get("duration")
            text = alignment.get("text")
            bounding_box = alignment.get("bounding_box")
            page_number = alignment.get("page_number")

            if duration is None or text is None or bounding_box is None:
                logging.error(f"Alignment data for chunk {i} is incomplete: {alignment}")
                raise ValueError(f"Alignment data for chunk {i} is incomplete.")

            combined_alignments.append({
                "begin": str(current_time_offset),
                "end": str(current_time_offset + duration),
                "text": text,
                "bounding_box": bounding_box,
                "page_number": page_number  # Include 'page_number'
            })
            current_time_offset += duration

        # Combine all audio segments
        logging.info("Combining audio segments...")
        combined_audio = AudioSegment.empty()
        for audio_file_path in audio_segments:
            if not audio_file_path:
                logging.warning("Missing audio segment. Skipping.")
                continue
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