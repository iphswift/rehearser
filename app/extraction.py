import requests
from bs4 import BeautifulSoup
import re
import os

# PDF and text processing functions
def extract_text_from_pdf(pdf_file_path, xml_output_path):
    print("Extracting text from PDF...")
    with open(pdf_file_path, 'rb') as pdf_file:
        files = {'input': pdf_file}
        response = requests.post('http://grobid:8070/api/processFulltextDocument', files=files)
        if response.status_code == 200:
            with open(xml_output_path, 'w') as xml_file:
                xml_file.write(response.text)
            return response.text
        else:
            print(f"Failed to process the PDF. Status code: {response.status_code}")
            return None
        
# Gets body text divs from Grobid XML.
def extract_body_text(grobid_output):
    print("Extracting body text...")
    soup = BeautifulSoup(grobid_output, 'xml')
    body = soup.find('body')

    main_text = []

    if body:
        for div in body.find_all('div'):
            for element in div.find_all(['p', 'head']):
                # Skip reference divs.
                for ref in element.find_all('ref', {'type': 'bibr'}):
                    ref.decompose()
                text = decompose_ligatures_in_string(element.get_text())
                main_text.append(text.strip())

    return main_text

def decompose_ligatures_in_string(input_string):
    # Mapping of ligature ASCII/Unicode values to their corresponding decomposed characters
    ligature_map = {
        0xFB01: "fi",  # Unicode for fi ligature
        0xFB02: "fl",  # Unicode for fl ligature
        0xFB00: "ff",  # Unicode for ff ligature
        0xFB03: "ffi", # Unicode for ffi ligature
        0xFB04: "ffl", # Unicode for ffl ligature
    }

    for ligature, decomposed in ligature_map.items():
        input_string = input_string.replace(chr(ligature), decomposed)

    return input_string

# Removes text that are believed to be tables, figures, etc. based on simple regex.
# This is a naive approach and may not work for all cases.
def filter_non_expository(texts):
    print("Removing non-expository text...")

    expository_texts = []
    for text in texts:
        if not is_header_for_table_or_figure(text):
            expository_texts.append(text)

    return '\n'.join(expository_texts)

def is_header_for_table_or_figure(text):
    patterns = [
        r'^\s*(Table|Figure|Fig\.|Tbl\.)\s+\d+.*$',
        r'^\s*(Table|Figure|Fig\.|Tbl\.).+$'
    ]
    for pattern in patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    return False


# Extracts narrational text from a PDF file using Grobid
def get_narrational_text(pdf_file_path, base_name, output_dir, xml_output_path):
    narrational_text_path = os.path.join(output_dir, f"{base_name}_narrational_text.txt")
    
    if os.path.exists(narrational_text_path):
        with open(narrational_text_path, 'r') as narrational_text_file:
            narrational_text = narrational_text_file.read()
    else:
        if os.path.exists(xml_output_path):
            with open(xml_output_path, 'r') as xml_file:
                grobid_output = xml_file.read()
        else:
            grobid_output = extract_text_from_pdf(pdf_file_path, xml_output_path)
            if not grobid_output:
                raise Exception("Failed to extract text from PDF.")
                
        body_text = extract_body_text(grobid_output)
        narrational_text = filter_non_expository(body_text)
                
        # Write the narrational text to a file
        with open(narrational_text_path, 'w') as narrational_text_file:
            narrational_text_file.write(narrational_text)
    
    return narrational_text