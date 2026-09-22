from chunking import create_chunks
from ocr import extract_text
from vector_store import VectorStore

if __name__ == "__main__":
    documents_path = "../dataset/Resume_dataset"
    # Extraction sur chaque image
    ocr_dict = extract_text(documents_path)
    # Création des chunks
    created_chunks = create_chunks(ocr_dict)
    # Initialisation de la base vectorielle
    db = VectorStore()
    # Remplissage de la base vectorielle
    db.add_chunks(created_chunks)
