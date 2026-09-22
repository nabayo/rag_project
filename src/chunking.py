from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(ocr_dict, chunk_size=100, chunk_overlap=20):
    """
    Création des chunks en utilisant langchain RecursiveCharacter.

    Parameters
    ----------
    ocr_dict : dict
        Dictionnaire contenant le texte extrait de chaque document.
    chunk_size : int, optional
        Taille de chaque chunk. Par défaut fixé à 100.
    chunk_overlap : int, optional
        Taille de l'overlap des chunk. Par défaut fixé à 20.

    Returns
    ----------
    all_chunks : list
        Liste contenant tous les chunks créés.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20, length_function=len
    )
    all_chunks = []
    for k, txt in ocr_dict.items():
        chunks = splitter.create_documents(
            texts=[txt],
            metadatas=[{"source": k}],
        )
        all_chunks.extend(chunks)

    print(
        f"Nombre total de chunks générés pour tous les documents : {len(all_chunks)}"
    )
    return all_chunks
