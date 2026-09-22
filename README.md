# Petit projet réalisé pour une prise en main de sujets relatifs aux RAG, Chatbot et LLM.

Réalisé par moi ! :)

Sujet du projet :

À partir d'images de cvs, construire une chatbot basé sur une architecture RAG.

## Extraction des informations contenues dans les images avec EasyOCR

Fichier ```ocr.py```

## Chunking avec LangChain

Fichier ```chunking.py```

## Création de la base vectorielle

Fichier ```vector_store.py```

## Lancement

Fichier ```ingest.py```

## RAG

Fichier ```gen.py```

## Interface streamlit pour le chatbot

Fichier ```app.py```


Installation et lancement de ollama (LLM utilisé)
```
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama run llama3.2
```