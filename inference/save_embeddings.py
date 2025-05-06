from RecRAG import RecRag

recRAG = RecRag(
    "SeppeV/JokeTailor_big_set_annotated",
    "SeppeV/RatedDatasetJokeTailor",
    "SeppeV/JokeTailorBiEncoder",
    "SeppeV/JokeTailorCrossEncoder",
    "jokeText",
    best_k=7
)

recRAG.save_item_embeddings(
    "jokeTailor_embeddings.pkl",
    repo_name="SeppeV/jokeTailor_embeddings",
    make_repo=False,
)

recRAG.save_user_embeddings("SeppeV/jokeTailor_user_embeddings")
