import os
import re
import random
import numpy as np
import pandas as pd 
import pytesseract  
from PIL import Image
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import streamlit as st 

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to your image file
image_path = r'C:\Users\Prakash Humbe\OneDrive\Desktop\Bharat Fellowship Project\Images\R106W92ApplicationForm-1.png'

# Function to extract text from an image
def extract_text_from_image(image_path):
    
    try:
        img = Image.open(image_path)  # Open the image
        text = pytesseract.image_to_string(img)  # Extract text using OCR
        return text
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Test the function
extracted_text = extract_text_from_image(image_path)
print("Extracted Text:\n", extracted_text) 

def create_simulated_dataset(num_samples=100): 
    application_texts = [
        "This is an application for a new passport. Please process my application at the earliest.",
        "Application for a driving license renewal. Attached are the required documents."
    ]
    form_texts = [
        "Please fill out this form with your personal details and submit for processing.",
        "The form includes fields such as name, address, and contact number. Ensure all fields are completed."
    ]
    report_texts = [
        "This report outlines the quarterly financial performance and expenditure details.",
        "Annual report submitted by the department summarizing all key activities and outcomes."
    ]
    
    texts = []
    labels = [] 
    for _ in range(num_samples):
        doc_type = random.choice(["Application", "Form", "Report"])
        if doc_type == "Application":
            texts.append(random.choice(application_texts))
        elif doc_type == "Form":
            texts.append(random.choice(form_texts))
        else:
            texts.append(random.choice(report_texts))
        labels.append(doc_type)
    
    df = pd.DataFrame({"text": texts, "label": labels})
    return df 

    # Text Preprocessing Function
# ---------------------------
def preprocess_text(text):
    # Convert text to lower case
    text = text.lower()
    # Remove non-alphanumeric characters (keep basic punctuation)
    text = re.sub(r'[^a-z0-9\s\.,]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text 

    # Load the BERT tokenizer and model for sequence classification
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3) 

def tokenize_function(examples): 
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Simulate the dataset
df = create_simulated_dataset(num_samples=200)
# Preprocess the text column
df["text"] = df["text"].apply(preprocess_text)

# Map string labels to integers
label_mapping = {"Application": 0, "Form": 1, "Report": 2}
df["label_id"] = df["label"].map(label_mapping)

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create a Hugging Face Dataset from the pandas DataFrame (using simple list conversion)
train_dataset = [
    {"text": row["text"], "label": row["label_id"]} for _, row in train_df.iterrows()
]
test_dataset = [
    {"text": row["text"], "label": row["label_id"]} for _, row in test_df.iterrows()
]

# Tokenize the datasets
# For simplicity, we use a list comprehension; in practice, you might use the datasets library.
train_encodings = [tokenizer(sample["text"], truncation=True, padding="max_length", max_length=128) for sample in train_dataset]
test_encodings = [tokenizer(sample["text"], truncation=True, padding="max_length", max_length=128) for sample in test_dataset] 

# Convert tokenized texts and labels to PyTorch tensors
class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
      
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.batchify(self.encodings).items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    def batchify(self, encodings):
        # Convert list of dicts to dict of lists
        batch = {}
        for enc in encodings:
            for key, value in enc.items():
                batch.setdefault(key, []).append(value)
        return batch

train_labels = [sample["label"] for sample in train_dataset]
test_labels = [sample["label"] for sample in test_dataset]

train_dataset_torch = DocumentDataset(train_encodings, train_labels)
test_dataset_torch = DocumentDataset(test_encodings, test_labels) 

def compute_metrics(pred): 
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define training arguments for the Hugging Face Trainer
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=10,                 # number of warmup steps for learning rate scheduler
    evaluation_strategy="epoch",     # evaluate at the end of each epoch
    save_strategy="epoch",           # save model at the end of each epoch
    logging_dir="./logs",            # directory for storing logs
    logging_steps=5,
    load_best_model_at_end=True,
)


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_torch,
    eval_dataset=test_dataset_torch,
    compute_metrics=compute_metrics,
)  
# Create a mapping of document classification labels to government departments or workflow actions.
routing_dict = {
    0: "Department of Applications",   # For 'Application' documents
    1: "Forms Processing Unit",          # For 'Form' documents
    2: "Reporting and Analysis Division" # For 'Report' documents
} 

def get_workflow_routing(prediction_label): 
    return routing_dict.get(prediction_label, "Document Checking Department") 

def run_dashboard(): 
    st.title("Automated Document Classification and Workflow Routing System")
    
    st.write("Upload a document image or a text file to get started.")
    
    # File uploader: allow image (for OCR) or text file uploads
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "txt"])
    
    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
        st.write("File Details:", file_details)
        
        # If the file is an image, perform OCR; if text, read the content directly.
        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            # Save the uploaded file temporarily
            # TODO: Replace the temporary file path if needed.
            with open("temp_uploaded_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image("temp_uploaded_image.jpg", caption="Uploaded Document", use_column_width=True)
            
            # Extract text from the image using OCR
            extracted_text = extract_text_from_image("temp_uploaded_image.jpg")
            st.subheader("Extracted Text from Image:")
            st.write(extracted_text)
            document_text = extracted_text
        else:
            # Read the content of the text file
            document_text = uploaded_file.read().decode("utf-8")
            st.subheader("Document Text:")
            st.write(document_text)
        
        # Preprocess the document text
        processed_text = preprocess_text(document_text)
        
        # Tokenize the input text for classification
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        
        # Get model prediction (ensure model is in evaluation mode)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        
        # Map prediction to human-readable label (reverse mapping from label_mapping)
        inv_label_mapping = {v: k for k, v in label_mapping.items()}
        predicted_label = inv_label_mapping.get(prediction, "Unknown")
        st.subheader("Predicted Document Type:")
        st.write(predicted_label)
        
        # Get workflow routing decision based on prediction
        routing_decision = get_workflow_routing(prediction)
        st.subheader("Recommended Workflow Routing:")
        st.write(routing_decision)
        
        # Optionally, clean up temporary files
        if os.path.exists("temp_uploaded_image.jpg"):
            os.remove("temp_uploaded_image.jpg") 
        
def evaluate_model(): 

    model.eval()
    all_preds = []
    all_labels = []
    for i in range(len(test_dataset_torch)):
        sample = test_dataset_torch[i]
        # Add batch dimension
        input_ids = sample["input_ids"].unsqueeze(0)
        attention_mask = sample["attention_mask"].unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        all_preds.append(pred)
        all_labels.append(sample["labels"].item())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    print("Evaluation Metrics on Test Dataset:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}") 

   