import os

import easyocr
from tqdm import tqdm


def extract_text(path):
    """
    Extraction du texte de tous les fichiers d'un répertoire en utilisant
    l'OCR.

    Parameters
    ----------
    path : str
        Chemin du répertoire contenant les fichiers.

    Returns :
    ocr_dict : dict
        Dictionnaire avec pour clé le nom du fichier et pour
        valeur le texte extrait.
    """
    # Initialisation
    reader = easyocr.Reader(["en"])
    files = sorted(os.listdir(path))
    ocr_dict = {}
    # Parcours des fichiers et OCR sur chacun d'eux
    for file in tqdm(files, desc="Extraction par OCR"):
        img_path = os.path.join(path, file)
        results = reader.readtext(img_path, detail=0)

        # Affiche le texte extrait
        extracted_text = "\n".join(results)
        ocr_dict[file] = extracted_text
    return ocr_dict
