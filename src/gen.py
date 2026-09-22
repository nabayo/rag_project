import ollama


def build_prompt(context, question):
    """
    Construction du prompt pour le LLM.

    Parameters
    ----------
    context : str
        Contexte de la question.
    question : str
        Queston de l'utilisateur.

    Returns
    ----------
    str
        Le prompt formaté pour le LLM.
    """
    return f"""
            You are a helpful assistant to give informations about resumes provided.
            You should respond in the language of the question. You should always
            be polite.
            If you don't know the answer or if it's not present in the context,
            you should state that you do not have the information.
            Context:
            {context}
            Question:
            {question}
            """


def ask_rag(db, encoder, question, top_k=5):
    """
    Construction du prompt pour le LLM.

    Parameters
    ----------
    db : VectorStore
        Base de données vectorielle.
    encoder :
        Modèle de reranking utilisé.
    question : str
        Queston de l'utilisateur.
    top_k : int, optional
        Nombre de chunks à utiliser pour générer la réponse. Par défaut fixé à 5.

    Returns
    ----------
    dict :
        Réponse du modèle, sources et scores.
    """

    # Récupération des chunks pertinents
    retrieved_chunks = db.retrieval_with_reranker(
        encoder, question, top_k=top_k
    )

    # Construction du contexte
    context_blocks = [
        f"--- Source: {chunk['metadata'].get('source')} (Page {chunk['metadata'].get('page')}) ---\n{chunk['content']}"
        for chunk in retrieved_chunks
    ]
    context = "\n\n".join(context_blocks)

    # Construiction du prompt
    prompt = build_prompt(context, question)

    response = ollama.chat(
        model="llama3.2", messages=[{"role": "user", "content": prompt}]
    )

    # Retourner la réponse du modèle, les sources et scores
    return {
        "answer": response["message"]["content"],
        "sources": [
            (chunk["metadata"], chunk["score"]) for chunk in retrieved_chunks
        ],
    }
