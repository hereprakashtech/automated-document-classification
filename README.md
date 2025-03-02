# automated-document-classification
Automated Document Classification and Workflow Routing System
Project Overview
The Automated Document Classification and Workflow Routing System is an AI-powered application that classifies documents into predefined categories (e.g., applications, forms, reports) and recommends appropriate processing workflows. The system utilizes OCR (Optical Character Recognition) to extract text from scanned documents and BERT-based NLP models for classification.

Key Features
✅ OCR-Based Text Extraction – Uses Tesseract OCR to extract text from images.
✅ Document Classification – Leverages a fine-tuned BERT model to classify documents.
✅ Workflow Routing – Assigns documents to relevant departments based on classification.
✅ Streamlit UI – Provides an interactive web interface for easy document upload and classification.

Technologies Used
Python (Main programming language)
Tesseract OCR (For extracting text from images)
BERT (Transformers) (For text classification)
Scikit-Learn (For dataset splitting and evaluation)
Hugging Face Transformers (For NLP model training)
PyTorch (For deep learning model implementation)
Streamlit (For creating an interactive web app)
GitHub (For version control and code hosting)
Installation & Setup
1. Clone the Repository
sh
Copy
Edit
git clone https://github.com/your-username/automated-document-classification.git
cd automated-document-classification
2. Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit App
sh
Copy
Edit
streamlit run "Automated Document Classification and Workflow Routing System.py"
How It Works
Upload a document image (JPEG, PNG) or a text file.
OCR extracts text from image-based documents.
The BERT model classifies the document into one of three categories:
Application
Form
Report
The system recommends workflow routing based on classification.
Example Output
yaml
Copy
Edit
Extracted Text: "This is an application for a new passport. Please process my request."
Predicted Document Type: Application
Recommended Workflow Routing: Department of Applications
Future Enhancements
🔹 Expand document classification categories.
🔹 Improve OCR accuracy using deep learning-based text recognition.
🔹 Implement multi-language support.
