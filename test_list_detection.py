#!/usr/bin/env python3
"""
Simple test script to check list detection in a PDF
"""
import fitz
import re

LIST_PATTERN_SIMPLE = re.compile(r"^(\d+\.|•|-)\s+")
LIST_PATTERN_COMPLEX = re.compile(r'^\s*(\d+\.|\d+\)|\d+\-|[a-zA-Z]\.|[a-zA-Z]\)|[a-zA-Z]\-|\•|\-|\*)\s')

def is_list_item_simple(text):
    return bool(LIST_PATTERN_SIMPLE.match(text.strip()))

def is_list_item_complex(text):
    return bool(LIST_PATTERN_COMPLEX.match(text.strip()))

def test_pdf_lists(pdf_path):
    doc = fitz.open(pdf_path)
    print(f"Testing PDF: {pdf_path}")
    print(f"Pages: {len(doc)}")

    for page_number, page in enumerate(doc):
        print(f"\n--- Page {page_number} ---")
        text_dict = page.get_text("dict")

        for block in text_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    spans = line["spans"]
                    if not spans:
                        continue

                    # Combine spans in this line
                    line_text = "".join(span["text"] for span in spans)
                    line_text = line_text.strip()

                    if line_text:
                        is_list_simple = is_list_item_simple(line_text)
                        is_list_complex = is_list_item_complex(line_text)
                        if is_list_simple or is_list_complex:
                            marker = "LIST"
                            if is_list_simple and not is_list_complex:
                                marker += "(simple)"
                            elif is_list_complex and not is_list_simple:
                                marker += "(complex)"
                            else:
                                marker += "(both)"
                        else:
                            marker = "TEXT"
                        print(f"[{marker}] '{line_text}'")

if __name__ == "__main__":
    test_pdf_lists("/home/joulifestyle/Downloads/purposes.pdf")