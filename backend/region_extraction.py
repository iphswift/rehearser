from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json
import re
import pdfplumber  # Ensure you have pdfplumber installed: pip install pdfplumber
import os
import logging

# Determine the directory where the log file should be placed
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdf_segmentation.log')
log_dir = os.path.dirname(log_file)

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class TextSegment:
    id: int  # Unique identifier for the segment
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, top, x1, bottom)


@dataclass
class Adjacency:
    left: List[int] = field(default_factory=list)   # IDs of segments to the left
    right: List[int] = field(default_factory=list)  # IDs of segments to the right
    top: List[int] = field(default_factory=list)    # IDs of segments above
    bottom: List[int] = field(default_factory=list) # IDs of segments below


@dataclass
class PageData:
    page_number: int
    segments: List[TextSegment]
    adjacencies: Dict[int, Adjacency]
    connected_components: List[List[int]] = field(default_factory=list)  # Connected components


def is_close(a: float, b: float, tolerance: float) -> bool:
    """Check if two float numbers are within a certain tolerance."""
    return abs(a - b) <= tolerance


def group_words_into_lines(words: List[dict], y_tolerance: float = 2.0) -> List[List[dict]]:
    """
    Groups words into lines based on their vertical positions with a specified tolerance.
    """
    lines = []
    sorted_words = sorted(words, key=lambda w: w['top'])  # Sort by y-coordinate

    for word in sorted_words:
        placed = False
        for line in lines:
            # Check if the word's y-coordinate is close to the current line's y-coordinate
            # Using the average y-coordinate of the line for comparison
            line_avg_y = sum(w['top'] for w in line) / len(line)
            if is_close(word['top'], line_avg_y, y_tolerance):
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])
    return lines


def segment_line_into_columns(line_words: List[dict], gap_threshold_ratio: float = 0.5) -> List[List[dict]]:
    """
    Segments a line into columns based on gaps between words.
    """
    if not line_words:
        return []

    # Sort words by their x0 coordinate
    sorted_words = sorted(line_words, key=lambda w: w['x0'])

    # Calculate gaps between consecutive words
    gaps = []
    for i in range(1, len(sorted_words)):
        prev_word = sorted_words[i - 1]
        current_word = sorted_words[i]
        gap = current_word['x0'] - prev_word['x1']
        gaps.append(gap)

    if not gaps:
        return [sorted_words]

    average_gap = sum(gaps) / len(gaps)
    threshold = average_gap * gap_threshold_ratio

    # Segment based on threshold
    segments = []
    current_segment = [sorted_words[0]]
    for i in range(1, len(sorted_words)):
        gap = gaps[i - 1]
        if gap > threshold:
            segments.append(current_segment)
            current_segment = [sorted_words[i]]
        else:
            current_segment.append(sorted_words[i])
    segments.append(current_segment)

    return segments


def calculate_bounding_box(words: List[dict]) -> Tuple[float, float, float, float]:
    """
    Calculates the bounding box that encompasses all words in the list.
    """
    x0 = min(word['x0'] for word in words)
    top = min(word['top'] for word in words)
    x1 = max(word['x1'] for word in words)
    bottom = max(word['bottom'] for word in words)
    return (x0, top, x1, bottom)


def find_adjacencies(segments: List[TextSegment], proximity: float = 5.0) -> Dict[int, Adjacency]:
    """
    For each segment, find adjacent segments in four directions: left, right, top, bottom.
    """
    adjacencies = {segment.id: Adjacency() for segment in segments}

    for seg_a in segments:
        a_x0, a_top, a_x1, a_bottom = seg_a.bbox

        for seg_b in segments:
            if seg_a.id == seg_b.id:
                continue  # Skip self

            b_x0, b_top, b_x1, b_bottom = seg_b.bbox

            # Check left adjacency
            if is_close(a_x0, b_x1, proximity) and (a_top < b_bottom and a_bottom > b_top):
                adjacencies[seg_a.id].left.append(seg_b.id)

            # Check right adjacency
            if is_close(a_x1, b_x0, proximity) and (a_top < b_bottom and a_bottom > b_top):
                adjacencies[seg_a.id].right.append(seg_b.id)

            # Check top adjacency
            if is_close(a_top, b_bottom, proximity) and (a_x0 < b_x1 and a_x1 > b_x0):
                adjacencies[seg_a.id].top.append(seg_b.id)

            # Check bottom adjacency
            if is_close(a_bottom, b_top, proximity) and (a_x0 < b_x1 and a_x1 > b_x0):
                adjacencies[seg_a.id].bottom.append(seg_b.id)

    return adjacencies


def find_connected_components(adjacencies: Dict[int, Adjacency]) -> List[List[int]]:
    """
    Finds connected components in the adjacency graph.
    """
    visited = set()
    connected_components = []

    def dfs(node_id, component):
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.append(current)
                neighbors = (
                    adjacencies[current].left +
                    adjacencies[current].right +
                    adjacencies[current].top +
                    adjacencies[current].bottom
                )
                stack.extend(neighbors)

    for segment_id in adjacencies.keys():
        if segment_id not in visited:
            component = []
            dfs(segment_id, component)
            connected_components.append(component)

    return connected_components


def extract_connected_regions_from_pdf(
    pdf_path: str,
    gap_threshold_ratio: float = 1.5,
    y_tolerance: float = 2.0,
    proximity: float = 5.0
) -> Tuple[List[PageData], Dict[int, Tuple[int, Tuple[float, float, float, float]]], Dict[int, int]]:
    """
       Segments a PDF into lines and columns, maps bounding boxes to their immediate neighbors,
    and finds connected components.

    Additionally, returns a mapping from unique segment ID to a tuple containing:
    - page number
    - bounding box

    :param pdf_path: Path to the input PDF file.
    :param gap_threshold_ratio: Ratio to determine column segmentation based on gaps.
    :param y_tolerance: Tolerance for grouping words into the same line based on vertical position.
    :param proximity: Proximity threshold for determining adjacency.
    :return: 
        - List of PageData objects containing segmentation and adjacency information.
        - Dictionary mapping unique segment_id to (page_number, bounding_box).
        - Dictionary mapping unique segment_id to component_id.
    """
    logging.info(f"Opening PDF file: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_pages_data = []
            segment_id_to_info = {}
            segment_id_to_component_id = {}
            global_segment_id = 0  # Initialize a global segment ID
            global_component_id = 0  # Initialize a global component ID

            for page_num, page in enumerate(pdf.pages, start=1):
                logging.info(f"Processing Page {page_num}")
                words = page.extract_words(use_text_flow=False, keep_blank_chars=True, x_tolerance=1)
                if not words:
                    logging.warning(f"No words found on Page {page_num}.")
                    all_pages_data.append(PageData(
                        page_number=page_num,
                        segments=[],
                        adjacencies={},
                        connected_components=[]
                    ))
                    continue

                lines = group_words_into_lines(words, y_tolerance)
                logging.debug(f"Page {page_num}: Grouped into {len(lines)} lines.")
                page_segments = []

                for line_num, line in enumerate(lines, start=1):
                    sorted_line = sorted(line, key=lambda w: w['x0'])
                    segments = segment_line_into_columns(sorted_line, gap_threshold_ratio)
                    logging.info(f"Page {page_num}, Line {line_num}: Split into {len(segments)} columns.")

                    for seg in segments:
                        segment_text = ' '.join(word['text'] for word in seg)
                        bbox = calculate_bounding_box(seg)
                        text_segment = TextSegment(
                            id=global_segment_id,  # Assign unique ID
                            text=segment_text,
                            bbox=bbox
                        )
                        page_segments.append(text_segment)
                        # Map unique segment ID to (page number, bounding box)
                        segment_id_to_info[global_segment_id] = (page_num, bbox)
                        # Temporarily assign component_id as None
                        segment_id_to_component_id[global_segment_id] = None
                        logging.info(f"Segment ID {global_segment_id}: '{segment_text}' with bbox {bbox}")
                        global_segment_id += 1  # Increment global ID

                adjacencies = find_adjacencies(page_segments, proximity)
                logging.debug(f"Page {page_num}: Found adjacencies for {len(adjacencies)} segments.")
                connected_components = find_connected_components(adjacencies)
                logging.info(f"Page {page_num}: Identified {len(connected_components)} connected components.")

                # Assign unique component IDs and update the mapping
                for component in connected_components:
                    for seg_id in component:
                        segment_id_to_component_id[seg_id] = global_component_id
                    global_component_id += 1  # Increment for next component

                all_pages_data.append(PageData(
                    page_number=page_num,
                    segments=page_segments,
                    adjacencies=adjacencies,
                    connected_components=connected_components
                ))

            logging.info("Completed extraction of connected regions from PDF.")
            return all_pages_data, segment_id_to_info, segment_id_to_component_id

    except Exception as e:
        logging.error(f"An error occurred while processing the PDF: {e}", exc_info=True)
        return [], {}, {}


def sort_segments(pages: List[PageData]) -> List[List[TextSegment]]:
    """
    Sorts segments by connected components and reading order across all pages,
    prioritizing column-wise reading for multi-column layouts.

    :param pages: List of PageData objects.
    :return: List of lists, where each sublist contains TextSegments of a connected component.
    """
    sorted_components_sorted_segments = []

    for page in pages:
        id_to_segment = {segment.id: segment for segment in page.segments}

        # Step 1: Determine column boundaries
        # Extract all left positions
        left_positions = sorted(set(segment.bbox[0] for segment in page.segments))
        
        # Define a threshold for column grouping (e.g., 50 pixels)
        column_threshold = 50
        columns = []
        current_column = []

        for left in left_positions:
            if not current_column:
                current_column.append(left)
            elif left - current_column[-1] > column_threshold:
                columns.append(current_column)
                current_column = [left]
            else:
                current_column.append(left)
        if current_column:
            columns.append(current_column)

        # Determine the average left position for each column
        column_positions = [sum(col) / len(col) for col in columns]

        # Step 2: Assign each segment to a column
        def assign_column(segment: TextSegment) -> float:
            segment_left = segment.bbox[0]
            # Find the column with the closest left position
            closest_column = min(column_positions, key=lambda col_left: abs(col_left - segment_left))
            return closest_column

        segment_to_column = {segment.id: assign_column(segment) for segment in page.segments}

        # Step 3: Sort connected components based on their assigned columns
        def component_sort_key(component: List[int]) -> Tuple[float, int]:
            # Get the minimum column position in the component
            min_column = min(segment_to_column[seg_id] for seg_id in component)
            # Get the minimum top position in the component
            min_top = min(id_to_segment[seg_id].bbox[1] for seg_id in component)
            return (min_column, min_top)

        sorted_components = sorted(page.connected_components, key=component_sort_key)

        # Step 4: Sort segments within each component based on their position within the column
        for component in sorted_components:
            segments = [id_to_segment[seg_id] for seg_id in component]
            # Sort primarily by top position, then by left position within the column
            segments_sorted = sorted(segments, key=lambda seg: (seg.bbox[1], seg.bbox[0]))
            sorted_components_sorted_segments.append(segments_sorted)

    return sorted_components_sorted_segments

def word_based_lcs(text1: str, text2: str) -> int:
    """
    Computes the length of the Longest Common Subsequence (LCS) of words between two texts.
    
    :param text1: First text string.
    :param text2: Second text string.
    :return: Length of the LCS.
    """
    words1 = text1.split()
    words2 = text2.split()
    m, n = len(words1), len(words2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m):
        for j in range(n):
            if words1[i].lower() == words2[j].lower():
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    return dp[m][n]

def word_based_lcs_alignment_with_pairs(text1: str, text2: str):
    """
    Performs a word-based LCS. Returns a list of (i, j) pairs where:
      - i is the index of the matched word in text1
      - j is the index of the matched word in text2
    """
    words1 = text1.split()
    words2 = text2.split()
    m, n = len(words1), len(words2)

    dp = [[0]*(n+1) for _ in range(m+1)]
    # Fill dp matrix
    for i in range(m):
        for j in range(n):
            if words1[i].lower() == words2[j].lower():
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    # Backtrack to find the actual matched pairs
    pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        if words1[i-1].lower() == words2[j-1].lower():
            pairs.append((i-1, j-1))
            i -= 1
            j -= 1
        else:
            if dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1

    return list(reversed(pairs))


def match_chunk_sequentially(
    text_chunk: str,
    all_segments_in_reading_order: List[TextSegment],
    max_length_multiplier: float = 4,
    lcs_threshold: int = 1    # Minimum LCS length to consider a valid match
):
    """
    Returns:
      best_lcs_count: integer (max number of matched words)
      best_segments: list of TextSegment that gave that best LCS
    """
    logging.debug(f"Matching text_chunk: '{text_chunk}' with max_length_multiplier={max_length_multiplier} and lcs_threshold={lcs_threshold}")
    chunk_len = len(text_chunk)
    max_total_length = int(chunk_len * max_length_multiplier)
    logging.debug(f"Computed max_total_length: {max_total_length}")

    best_lcs_count = 0
    best_segments = []

    for start_idx in range(len(all_segments_in_reading_order)):
        current_segments = []
        current_text = ""
        current_lcs = 0

        # We try segments from start_idx onward
        for seg_idx in range(start_idx, len(all_segments_in_reading_order)):
            seg = all_segments_in_reading_order[seg_idx]
            new_text = f"{current_text} {seg.text}".strip() if current_text else seg.text

            # If the new_text is too long, stop
            if len(new_text) > max_total_length:
                logging.debug(f"New text length {len(new_text)} exceeds max_total_length {max_total_length}. Breaking.")
                break

            # Compute new LCS count
            new_lcs = word_based_lcs(text_chunk, new_text)
            logging.debug(f"Attempting to add Segment ID {seg.id}: '{seg.text}'. New LCS count: {new_lcs}")

            if new_lcs > current_lcs and new_lcs >= lcs_threshold:
                # improvement, so we accept this segment
                current_segments.append(seg)
                current_text = new_text
                current_lcs = new_lcs
                logging.debug(f"Accepted Segment ID {seg.id}. Updated LCS count: {current_lcs}")

                # also check if we beat the global best
                if current_lcs > best_lcs_count:
                    best_lcs_count = current_lcs
                    best_segments = current_segments.copy()
                    logging.debug(f"New best LCS count {best_lcs_count} with segments {[s.id for s in best_segments]}")
            else:
                # no improvement => stop concatenating from this start
                logging.debug(f"No improvement in LCS. Current LCS: {current_lcs}, New LCS: {new_lcs}. Breaking.")
                break

    logging.info(f"Best LCS count: {best_lcs_count} with segments {[s.id for s in best_segments]}")
    return best_lcs_count, best_segments

def build_segment_word_mapping(segments: List[TextSegment]) -> Tuple[str, List[int]]:
    """
    Concatenates all segment texts and maps each word to its segment index.

    :param segments: List of TextSegments.
    :return: Tuple of concatenated text and list mapping each word index to its segment index.
    """
    logging.debug("Building segment word mapping.")
    words = []
    owners = []
    for seg_idx, segment in enumerate(segments):
        seg_words = segment.text.split()
        words.extend(seg_words)
        owners.extend([seg_idx] * len(seg_words))
        logging.debug(f"Segment ID {segment.id}: '{segment.text}' split into words {seg_words}")
    concatenated_text = " ".join(words)
    logging.debug(f"Concatenated matched_text: '{concatenated_text}'")
    logging.debug(f"Owners list: {owners}")
    return concatenated_text, owners
    

def combine_bounding_boxes(bounding_boxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """
    Combines multiple bounding boxes into a single bounding box.

    :param bounding_boxes: List of bounding boxes, each represented as (x0, top, x1, bottom).
    :return: A single bounding box that encompasses all input bounding boxes.
    """
    if not bounding_boxes:
        return (0.0, 0.0, 0.0, 0.0)  # Return a default bounding box if input is empty

    x0 = min(box[0] for box in bounding_boxes)
    top = min(box[1] for box in bounding_boxes)
    x1 = max(box[2] for box in bounding_boxes)
    bottom = max(box[3] for box in bounding_boxes)

    return (x0, top, x1, bottom)


from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import logging

def match_text_to_regions_with_splitting(
    text_chunks: List[str],
    pages_data: List[PageData],
    segment_id_to_info: Dict[int, Tuple[int, Tuple[float, float, float, float]]],
    segment_id_to_component_id: Dict[int, int],
    max_workers: Optional[int] = None  # Optional parameter to specify number of threads
) -> List[dict]:
    """
    Matches text chunks to PDF regions with splitting based on connected components,
    ensuring that the final result is a flat list of dictionaries with consistent structure.

    :param text_chunks: List of text chunks to be matched.
    :param pages_data: List of PageData objects containing PDF segmentation information.
    :param segment_id_to_info: Dictionary mapping unique segment_id to (page_number, bounding_box).
    :param segment_id_to_component_id: Dictionary mapping unique segment_id to component_id.
    :param max_workers: Maximum number of threads to use. Defaults to number of processors on the machine.
    :return: List of dictionaries describing each matched narrational text block with page and combined bounding box info.
    """
    logging.info("Starting match_text_to_regions_with_splitting process.")
    
    # 1) Sort the PDF segments by connected component and reading order
    component_sorted_segments = sort_segments(pages_data)
    logging.debug(f"Sorted segments into {len(component_sorted_segments)} connected components.")

    # Initialize a dictionary to hold results with chunk indices
    indexed_results = {}

    # Define a helper function for processing a single text_chunk
    def process_chunk(chunk_index: int, text_chunk: str) -> List[dict]:
        logging.info(f"Processing text_chunk {chunk_index}: '{text_chunk}'")
        
        # Use match_chunk_sequentially to find the best sequence of segments for this chunk
        best_lcs_count, best_segments = match_chunk_sequentially(
            text_chunk,
            [seg for component in component_sorted_segments for seg in component]
        )

        if best_lcs_count > 0:  # Only proceed if there is a strong match
            # Use the correct mapping to determine unique connected components
            unique_components = set(segment_id_to_component_id[seg.id] for seg in best_segments)
            if len(unique_components) == 1:
                # Single component match, no need to split
                component_id = unique_components.pop()
                seg_ids = [seg.id for seg in best_segments]
                # Determine the page number and bounding boxes from the segments
                page_numbers = set()
                bounding_boxes = []
                for seg_id in seg_ids:
                    page_num, bbox = segment_id_to_info.get(seg_id, (None, None))
                    if page_num:
                        page_numbers.add(page_num)
                    if bbox:
                        bounding_boxes.append(bbox)
                page_num = list(page_numbers)[0] if len(page_numbers) == 1 else list(page_numbers)             
                combined_bbox = combine_bounding_boxes(bounding_boxes)  # Combine bounding boxes
                result = {
                    'narrational_text_block': text_chunk,
                    'page_number': page_num,
                    'bounding_box': combined_bbox  # Store combined bounding box
                }
                return [result]  # Return as a list for consistency
            else:
                # Multiple component match, split by component
                sub_chunks = split_chunk_by_components_if_needed(
                    text_chunk,
                    best_segments,
                    {seg.id: segment_id_to_component_id[seg.id] for seg in best_segments}
                )
                result = []
                for sub_chunk in sub_chunks:
                    # Extract relevant information from each sub_chunk
                    sub_text = sub_chunk['sub_chunk']
                    component_id = sub_chunk['component_id']

                    # Identify which segments correspond to this component_id
                    related_segments = [seg for seg in best_segments if segment_id_to_component_id[seg.id] == component_id]
                    seg_ids = [seg.id for seg in related_segments]
                    
                    # Determine the page number(s) and combined bounding box for these segments
                    page_numbers = set()
                    bounding_boxes = []
                    for seg_id in seg_ids:
                        page_num, bbox = segment_id_to_info.get(seg_id, (None, None))
                        if page_num:
                            page_numbers.add(page_num)
                        if bbox:
                            bounding_boxes.append(bbox)
                    page_num = list(page_numbers)[0] if len(page_numbers) == 1 else list(page_numbers)
                    combined_bbox = combine_bounding_boxes(bounding_boxes)
                    
                    # Append the sub_chunk as a regular chunk in the results
                    result.append({
                        'narrational_text_block': sub_text,
                        'page_number': page_num,
                        'bounding_box': combined_bbox
                    })
                return result  # Return as a list
        else:
            # No strong match found, we leave it unmatched
            result = {
                'narrational_text_block': text_chunk,
                'page_number': None,
                'bounding_box': (0.0, 0.0, 0.0, 0.0)  # Default bounding box for unmatched
            }
            return [result]  # Return as a list for consistency

    # 2) Use ThreadPoolExecutor to process text_chunks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and map futures to their chunk indices
        future_to_index = {
            executor.submit(process_chunk, idx, chunk): idx 
            for idx, chunk in enumerate(text_chunks)
        }
        
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                data = future.result()
                indexed_results[idx] = data  # Store the list of results for this chunk
            except Exception as exc:
                logging.error(f"Text chunk {idx} generated an exception: {exc}", exc_info=True)
                # Assign a default unmatched entry
                indexed_results[idx] = [{
                    'narrational_text_block': text_chunks[idx],
                    'page_number': None,
                    'bounding_box': (0.0, 0.0, 0.0, 0.0)
                }]

    # 3) Assemble the final results in the original order
    final_results = []
    for idx in range(len(text_chunks)):
        chunk_results = indexed_results.get(idx, [{
            'narrational_text_block': text_chunks[idx],
            'page_number': None,
            'bounding_box': (0.0, 0.0, 0.0, 0.0)
        }])
        final_results.extend(chunk_results)

    logging.info("Completed match_text_to_regions_with_splitting process.")
    return final_results


def split_text(text, max_length=222):
    logging.info(f"Splitting text into chunks with max_length={max_length}")
    sentences = re.split(r'(?<=[.!?,;:])\s+', text)
    logging.debug(f"Split into {len(sentences)} sentences.")
    chunks = []
    current_chunk = ""

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        logging.debug(f"Sentence {i}: '{sentence}' with {len(words)} words.")

        for word in words:
            if len(current_chunk) + len(word) + 1 <= max_length:
                current_chunk += (" " + word if current_chunk else word)
            else:
                chunks.append(current_chunk)
                logging.debug(f"Added chunk: '{current_chunk}'")
                current_chunk = word

        if i + 1 < len(sentences) and len(current_chunk) + len(sentences[i + 1]) + 1 > max_length:
            chunks.append(current_chunk)
            logging.debug(f"Added chunk due to next sentence length: '{current_chunk}'")
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)
        logging.debug(f"Added final chunk: '{current_chunk}'")

    logging.info(f"Total chunks created: {len(chunks)}")
    return chunks

def split_chunk_by_components_if_needed(
    text_chunk: str,
    best_segments: List[TextSegment],
    segment_id_to_component: Dict[int, int]
) -> List[dict]:
    """
    Splits a text chunk into subchunks based on connected components.

    Each subchunk is returned as a dictionary containing:
      - 'sub_chunk': The text of the subchunk.
      - 'component_id': The ID of the connected component it belongs to.
      - 'matched_indices': The word indices in the original text_chunk that were matched.

    :param text_chunk: The original text chunk to be split.
    :param best_segments: The list of TextSegments matched to this text chunk.
    :param segment_id_to_component: Mapping from segment ID to component ID.
    :return: List of subchunk dictionaries.
    """
    logging.debug(f"Entering split_chunk_by_components_if_needed with text_chunk: '{text_chunk}'")
    logging.debug(f"Number of best_segments: {len(best_segments)}")

    if not best_segments:
        logging.warning("No best_segments found for the given text_chunk.")
        return [{
            'sub_chunk': text_chunk,
            'component_id': None,
            'matched_indices': []
        }]

    # 1) Figure out all the unique component IDs
    try:
        comp_ids = set(segment_id_to_component[seg.id] for seg in best_segments)
        logging.debug(f"Unique component IDs in best_segments: {comp_ids}")
    except KeyError as e:
        logging.error(f"Segment ID {e} not found in segment_id_to_component mapping.")
        return [{
            'sub_chunk': text_chunk,
            'component_id': None,
            'matched_indices': []
        }]

    if len(comp_ids) <= 1:
        single_comp_id = list(comp_ids)[0] if comp_ids else None
        logging.info(f"All segments belong to a single component ID: {single_comp_id}")
        return [{
            'sub_chunk': text_chunk,
            'component_id': single_comp_id,
            'matched_indices': None
        }]

    # 2) Build `matched_text` + `owners` to track which segment each word belongs to
    try:
        matched_text, owners = build_segment_word_mapping(best_segments)
        logging.debug(f"Matched text: '{matched_text}'")
        logging.debug(f"Owners mapping: {owners}")
    except Exception as e:
        logging.error(f"Error in build_segment_word_mapping: {e}")
        return [{
            'sub_chunk': text_chunk,
            'component_id': None,
            'matched_indices': []
        }]

    # 3) Find LCS alignment pairs
    try:
        pairs = word_based_lcs_alignment_with_pairs(text_chunk, matched_text)
        logging.debug(f"LCS alignment pairs: {pairs}")
    except Exception as e:
        logging.error(f"Error in word_based_lcs_alignment_with_pairs: {e}")
        pairs = []

    if not pairs:
        logging.warning("No LCS alignment pairs found. Unable to split chunk based on components.")
        return [{
            'sub_chunk': text_chunk,
            'component_id': None,
            'matched_indices': []
        }]

    # 4) Group each chunk word index i by the connected component
    from collections import defaultdict
    comp_to_indices = defaultdict(list)

    for (chunk_i, matched_j) in pairs:
        try:
            seg_idx = owners[matched_j]  # which segment in best_segments
            seg_component_id = segment_id_to_component[best_segments[seg_idx].id]
            comp_to_indices[seg_component_id].append(chunk_i)
            logging.debug(f"Word index {chunk_i} in chunk mapped to component ID {seg_component_id}")
        except IndexError:
            logging.error(f"matched_j index {matched_j} out of range for owners list.")
        except KeyError:
            logging.error(f"Segment ID {best_segments[seg_idx].id} not found in segment_id_to_component mapping.")

    logging.debug(f"Component to chunk word indices mapping: {dict(comp_to_indices)}")

    # 5) Build sub-chunks from these indices
    words_in_chunk = text_chunk.split()
    result = []

    for c_id, indices in comp_to_indices.items():
        indices_sorted = sorted(indices)
        sub_chunk_words = [words_in_chunk[i] for i in indices_sorted if i < len(words_in_chunk)]
        sub_chunk_str = " ".join(sub_chunk_words)
        logging.debug(f"Creating sub_chunk '{sub_chunk_str}' for component ID {c_id}")

        result.append({
            'sub_chunk': sub_chunk_str,
            'component_id': c_id,
            'matched_indices': indices_sorted
        })

    logging.info(f"split_chunk_by_components_if_needed result: {result}")
    return result

def bound_to_nearest_valid(chunks):
    """
    Updates entries in `chunks` where `page_number` is None by assigning them the `page_number`
    and `bounding_box` of the nearest valid entry (either previous or next).

    :param chunks: List of dictionaries, each containing at least `page_number` and `bounding_box` keys.
    :return: Updated list of dictionaries.
    """
    # First pass: fill with the previous valid entry
    last_valid_page_number = None
    last_valid_bounding_box = None

    for entry in chunks:
        if entry['page_number'] is None:
            # Use the last valid page_number and bounding_box if available
            if last_valid_page_number is not None and last_valid_bounding_box is not None:
                entry['page_number'] = last_valid_page_number
                entry['bounding_box'] = last_valid_bounding_box
        else:
            # Update the last valid values
            last_valid_page_number = entry['page_number']
            last_valid_bounding_box = entry['bounding_box']

    # Second pass: fill with the next valid entry if still None
    next_valid_page_number = None
    next_valid_bounding_box = None

    for entry in reversed(chunks):
        if entry['page_number'] is None:
            # Use the next valid page_number and bounding_box if available
            if next_valid_page_number is not None and next_valid_bounding_box is not None:
                entry['page_number'] = next_valid_page_number
                entry['bounding_box'] = next_valid_bounding_box
        else:
            # Update the next valid values
            next_valid_page_number = entry['page_number']
            next_valid_bounding_box = entry['bounding_box']

    return chunks

    

def segment_pdf_with_narrational_text(
    pdf_path: str,
    narrational_text: str,
    output_file: Optional[str] = None,  # Optional parameter for output file path
    max_workers: Optional[int] = None
) -> List[dict]:
    """
    ... [existing docstring] ...
    """
    logging.info("Starting PDF segmentation with narrational text matching.")
    max_length = 222
    gap_threshold_ratio = 1.5
    y_tolerance = 2.0
    proximity = 5.0

    # 1. Split the narrational text into manageable chunks
    text_chunks = split_text(narrational_text, max_length)
    logging.info(f"Total text_chunks created: {len(text_chunks)}")

    # 2. Extract connected regions from the PDF and get the segment-to-page mapping
    pages_data, segment_id_to_info, segment_id_to_component_id = extract_connected_regions_from_pdf(
        pdf_path=pdf_path,
        gap_threshold_ratio=gap_threshold_ratio,
        y_tolerance=y_tolerance,
        proximity=proximity
    )
    logging.info(f"Extracted data for {len(pages_data)} pages.")

    # 3. Match text chunks to PDF regions with splitting and page info
    result = match_text_to_regions_with_splitting(
        text_chunks, 
        pages_data, 
        segment_id_to_info,
        segment_id_to_component_id,
        max_workers=max_workers
    )
    logging.info(f"Matched text chunks to PDF regions. Total matches: {len(result)}")

    # 4. (Optional) Write results to a JSON file
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.info(f"Results successfully written to {output_file}")
        except Exception as e:
            logging.error(f"Failed to write results to {output_file}: {e}")

    logging.info("Completed PDF segmentation with narrational text matching.")
    
    result = bound_to_nearest_valid(result)
    
    return result
