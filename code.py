import os
import glob
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
import fitz  # PyMuPDF

# Download NLTK data (if not already downloaded)
nltk.download('punkt')

# Function to read PDF files and extract text
def read_pdf_files(folder_path):
    data = []
    file_paths = glob.glob(os.path.join(folder_path, '*.pdf'))

    for file_path in file_paths:
        with fitz.open(file_path) as pdf_doc:
            text = ''
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                text += page.get_text()

            data.append(text)

    return data

# Function to tokenize text into sentences
def tokenize_text(text):
    return sent_tokenize(text)

# Function to preprocess text (you may need to enhance this based on your data)
def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = tokenize_text(text)

    # Combine sentences to form a single string
    processed_text = ' '.join(sentences)

    return processed_text

# Main function for clustering
# Main function for clustering
def cluster_documents(data, file_paths):
    # Preprocess the documents
    preprocessed_data = [preprocess_text(doc) for doc in data]

    # Vectorize the documents using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    X = vectorizer.fit_transform(preprocessed_data)

    # Calculate cosine similarity matrix
    cosine_similarities = cosine_similarity(X)

    # K-Means clustering
    num_clusters = 5

      # You can adjust this based on your data
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(cosine_similarities)

    # Print cluster assignments with file content type
    for i in range(num_clusters):
        print(f"Cluster {i + 1}:")
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        for idx in cluster_indices:
            file_name = os.path.basename(file_paths[idx])
            print(f"  - File: {file_name}, Content Type: {file_name.split('.')[-1]}")


# Example usage
folder_path = '' # Give your documents Folder location
file_paths = glob.glob(os.path.join(folder_path, '*.pdf'))
data = read_pdf_files(folder_path)
cluster_documents(data, file_paths)
