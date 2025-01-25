import unittest
import os
import sys
import statistics
import glob
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from collections import Counter

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from backend.region_extraction import extract_connected_regions_from_pdf, find_columns_for_page, sort_segments



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

class TestPDFMetrics(unittest.TestCase):
    """
    Unit tests for validating PDF metrics such as average words per segment,
    average word length, and most common column formats across specified pages.
    """

    def _compute_average_metrics(self, segments):
        """
        Helper method to compute the average number of words and average word length in a list of segments.

        Args:
            segments (list): List of segment objects containing text.

        Returns:
            tuple: (average_word_count, min_word_count, max_word_count, average_word_length, min_word_length, max_word_length)
        """
        total_words = 0
        word_counts = []
        word_lengths = []

        for segment in segments:
            words = segment.text.split()
            word_count = len(words)
            word_counts.append(word_count)
            total_words += word_count

            for word in words:
                word_length = len(word)
                word_lengths.append(word_length)

        if not word_counts:
            average_word_count = 0
            min_word_count = 0
            max_word_count = 0
        else:
            average_word_count = statistics.mean(word_counts)
            min_word_count = min(word_counts)
            max_word_count = max(word_counts)

        if not word_lengths:
            average_word_length = 0
            min_word_length = 0
            max_word_length = 0
        else:
            average_word_length = statistics.mean(word_lengths)
            min_word_length = min(word_lengths)
            max_word_length = max(word_lengths)

        return (average_word_count, min_word_count, max_word_count,
                average_word_length, min_word_length, max_word_length)

    def _test_average_metrics(self, pdf_path, column_format):
        """
        Generalized test method to assert average word count and average word length in specified pages,
        applying different rules based on the column format.

        Args:
            pdf_path (str): Path to the PDF file to be tested.
            column_format (str): The column format of the PDF ('one_column' or 'two_column').
        """
        # Extract connected regions from the PDF
        pages_data, _, _ = extract_connected_regions_from_pdf(
            pdf_path=pdf_path,
            page_slice=slice(1, 2)  # Adjust as needed for pages 2-5
        )

        all_segments = []
        for page in pages_data:
            all_segments.extend(page.segments)

        # Ensure there are segments to analyze
        self.assertTrue(
            len(all_segments) > 0,
            f"No segments found in pages 2-5 of the PDF '{pdf_path}'."
        )

        # Compute the average number of words and average word lengthp per segment
        (average_word_count, min_wc, max_wc,
         average_word_length, min_wl, max_wl) = self._compute_average_metrics(all_segments)

        # Print the metrics to the console
        print(f"PDF '{pdf_path}' ({column_format}):")
        print(f"  Average Word Count per Segment: {average_word_count:.2f}")
        print(f"  Word Count Range: {min_wc} - {max_wc}")
        print(f"  Average Word Length: {average_word_length:.2f}")
        print(f"  Word Length Range: {min_wl} - {max_wl}\n")

        # Define expected ranges based on column format
        if column_format == 'two_column':
            expected_wc_min = 8
            expected_wc_max = 13
            expected_wl_min = 3
            expected_wl_max = 15
        elif column_format == 'one_column':
            expected_wc_min = 12  # Example: higher word count for single column
            expected_wc_max = 20
            expected_wl_min = 3
            expected_wl_max = 15
        else:
            self.fail(f"Unknown column format '{column_format}' for PDF '{pdf_path}'.")

        # Assert that the average word count is within the specified range
        self.assertGreaterEqual(
            average_word_count,
            expected_wc_min,
            f"Average word count {average_word_count:.2f} is less than {expected_wc_min} in PDF '{pdf_path}' ({column_format}). "
            f"Word Count Range: {min_wc} - {max_wc}; "
            f"Average Word Length: {average_word_length:.2f}"
        )
        self.assertLessEqual(
            average_word_count,
            expected_wc_max,
            f"Average word count {average_word_count:.2f} exceeds {expected_wc_max} in PDF '{pdf_path}' ({column_format}). "
            f"Word Count Range: {min_wc} - {max_wc}; "
            f"Average Word Length: {average_word_length:.2f}"
        )

        # Assert that the average word length is within a reasonable range
        # Typical English word lengths range from 3 to 10 characters
        self.assertGreaterEqual(
            average_word_length,
            expected_wl_min,
            f"Average word length {average_word_length:.2f} is less than {expected_wl_min} in PDF '{pdf_path}' ({column_format}). "
            f"Word Length Range: {min_wl} - {max_wl}; "
            f"Average Word Count: {average_word_count:.2f}"
        )
        self.assertLessEqual(
            average_word_length,
            expected_wl_max,  # Adjusted upper bound based on column format
            f"Average word length {average_word_length:.2f} exceeds {expected_wl_max} in PDF '{pdf_path}' ({column_format}). "
            f"Word Length Range: {min_wl} - {max_wl}; "
            f"Average Word Count: {average_word_count:.2f}"
        )

    def test_all_pdfs_average_metrics(self):
        """
        Test the average number of words per segment and average word length for all PDFs
        in the '/test_pdfs' subdirectories ('one_column' and 'two_column').
        Applies different validation rules based on the column format.
        """
        # Define the subdirectories and their corresponding column formats
        column_subdirs = {
            'two_column': 'two_column',
            'one_column': 'one_column'
        }

        # Iterate over each column format subdirectory
        for subdir, column_format in column_subdirs.items():
            # Construct the path to the subdirectory
            subdir_path = os.path.join(project_root, 'test/test_pdfs', subdir)

            # Use glob to find all PDF files in the subdirectory
            pdf_files = glob.glob(os.path.join(subdir_path, '*.pdf'))

            # Ensure that there are PDF files to test
            self.assertTrue(
                len(pdf_files) > 0,
                f"No PDF files found in the directory '{subdir_path}'."
            )

            # Iterate over each PDF file and perform the test within a subTest
            for pdf_path in pdf_files:
                with self.subTest(pdf=pdf_path, format=column_format):
                    self._test_average_metrics(pdf_path=pdf_path, column_format=column_format)

if __name__ == '__main__':
    unittest.main()
