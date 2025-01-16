import unittest
import os
import sys
import statistics
import glob

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from backend.region_extraction import extract_connected_regions_from_pdf

class TestPDFMetrics(unittest.TestCase):
    """
    Unit tests for validating the average number of words per segment
    and average word length across specified pages in PDF documents.
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

    def _test_average_metrics(self, pdf_path):
        """
        Generalized test method to assert average word count and average word length in specified pages.

        Args:
            pdf_path (str): Path to the PDF file to be tested.
        """
        # Extract connected regions from the PDF
        pages_data, _, _ = extract_connected_regions_from_pdf(
            pdf_path=pdf_path
        )

        # Define the page indices to include (pages 2-5 correspond to indices 1-4)
        target_page_indices = slice(1, 5)  # Pages 2, 3, 4, 5

        all_segments = []
        for page in pages_data[target_page_indices]:
            all_segments.extend(page.segments)

        # Ensure there are segments to analyze
        self.assertTrue(
            len(all_segments) > 0,
            f"No segments found in pages 2-5 of the PDF '{pdf_path}'."
        )

        # Compute the average number of words and average word length per segment
        (average_word_count, min_wc, max_wc,
         average_word_length, min_wl, max_wl) = self._compute_average_metrics(all_segments)

        # Print the metrics to the console
        print(f"PDF '{pdf_path}':")
        print(f"  Average Word Count per Segment: {average_word_count:.2f}")
        print(f"  Word Count Range: {min_wc} - {max_wc}")
        print(f"  Average Word Length: {average_word_length:.2f}")
        print(f"  Word Length Range: {min_wl} - {max_wl}\n")

        # Assert that the average word count is within the specified range
        self.assertGreaterEqual(
            average_word_count,
            5,
            f"Average word count {average_word_count:.2f} is less than 5 in PDF '{pdf_path}'. "
            f"Word Count Range: {min_wc} - {max_wc}; "
            f"Average Word Length: {average_word_length:.2f}"
        )
        self.assertLessEqual(
            average_word_count,
            20,
            f"Average word count {average_word_count:.2f} exceeds 20 in PDF '{pdf_path}'. "
            f"Word Count Range: {min_wc} - {max_wc}; "
            f"Average Word Length: {average_word_length:.2f}"
        )

        # Assert that the average word length is within a reasonable range
        # Typical English word lengths range from 3 to 10 characters
        self.assertGreaterEqual(
            average_word_length,
            3,
            f"Average word length {average_word_length:.2f} is less than 3 in PDF '{pdf_path}'. "
            f"Word Length Range: {min_wl} - {max_wl}; "
            f"Average Word Count: {average_word_count:.2f}"
        )
        self.assertLessEqual(
            average_word_length,
            10,  # Corrected the upper bound from 15 to 10 as per the comment
            f"Average word length {average_word_length:.2f} exceeds 10 in PDF '{pdf_path}'. "
            f"Word Length Range: {min_wl} - {max_wl}; "
            f"Average Word Count: {average_word_count:.2f}"
        )

    def test_all_pdfs_average_metrics(self):
        """
        Test the average number of words per segment and average word length for all PDFs
        in the '/test_pdfs' subdirectory across pages 2-5.
        """
        # Construct the path to the '/test_pdfs' directory
        test_pdfs_dir = os.path.join(project_root, 'test/test_pdfs')

        # Use glob to find all PDF files in the directory
        pdf_files = glob.glob(os.path.join(test_pdfs_dir, '*.pdf'))

        # Ensure that there are PDF files to test
        self.assertTrue(
            len(pdf_files) > 0,
            f"No PDF files found in the directory '{test_pdfs_dir}'."
        )

        # Iterate over each PDF file and perform the test within a subTest
        for pdf_path in pdf_files:
            with self.subTest(pdf=pdf_path):
                self._test_average_metrics(pdf_path=pdf_path)

if __name__ == '__main__':
    unittest.main()
