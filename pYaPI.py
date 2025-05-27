import os
import io
from google.cloud import vision
from google.api_core.exceptions import GoogleAPICallError
import fitz  # PyMuPDF
from PIL import Image


def setup_environment():
    """Set up the Google Cloud credentials."""
    creds_path = "C:/Users/mghir/Desktop/vision-458206-e466edc966f8.json"

    if not os.path.exists(creds_path):
        raise FileNotFoundError(f"Credentials file not found at: {creds_path}")

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path


def extract_text_from_pdf(pdf_path):
    """Converts PDF pages to images and extracts text using Vision API."""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        client = vision.ImageAnnotatorClient()
        doc = fitz.open(pdf_path)

        all_text = ""

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            pix = page.get_pixmap(dpi=300)  # High resolution for better OCR
            img_bytes = pix.tobytes("png")

            image = vision.Image(content=img_bytes)

            # Use DOCUMENT_TEXT_DETECTION for better structured text extraction
            response = client.document_text_detection(image=image)

            if response.error.message:
                raise GoogleAPICallError(response.error.message)

            texts = response.full_text_annotation.text
            print(f"\n--- Page {page_number + 1} ---")
            print(texts.strip())
            all_text += f"\n--- Page {page_number + 1} ---\n{texts.strip()}"

        return all_text

    except GoogleAPICallError as e:
        print(f"Google Cloud API Error: {e.message}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return None


if __name__ == "__main__":
    try:
        setup_environment()
        pdf_path = 'C:/Users/mghir/Downloads/TR2.pdf'
        print(f"Processing PDF for OCR: {pdf_path}")

        extracted_text = extract_text_from_pdf(pdf_path)

        if not extracted_text:
            print("No text detected or an error occurred.")
        else:
            # Optional: Save to a .txt file
            with open("extracted_text.txt", "w", encoding="utf-8") as f:
                f.write(extracted_text)

    except Exception as e:
        print(f"Setup error: {str(e)}")
