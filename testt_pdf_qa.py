import requests
import os
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/upload_and_query"
PDF_PATH = "sample.pdf"  # Path to your PDF file
QUESTION = "What is the capital of France?"  # QA query

def test_pdf_qa_endpoint():
    """
    Test the /upload_and_query endpoint for PDF QA.
    """
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file '{PDF_PATH}' not found.")
        return

    # Prepare the multipart/form-data request
    with open(PDF_PATH, "rb") as pdf_file:
        files = {
            "file": (os.path.basename(PDF_PATH), pdf_file, "application/pdf")
        }
        data = {
            "question": QUESTION
        }

        try:
            # Send POST request to the endpoint
            response = requests.post(API_URL, files=files, data=data, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Print the response
            print(f"Status Code: {response.status_code}")
            print("Response:", response.json())

        except requests.exceptions.RequestException as e:
            print(f"Error testing endpoint: {e}")
            if hasattr(e.response, "text"):
                print(f"Server Response: {e.response.text}")

if __name__ == "__main__":
    test_pdf_qa_endpoint()
