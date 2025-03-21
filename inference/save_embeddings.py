from RecRAG import RecRag

recRAG = RecRag(
    "SeppeV/JokeTailor_big_set_annotated",
    "SeppeV/rated_ds_test",
    "all-roberta-large-v1",
    "SeppeV/cross_encoder_test",
    "jokeText",
    best_k=7
)

recRAG.save_item_embeddings(
    "jokeTailor_embeddings.pkl",
    repo_name="SeppeV/jokeTailor_embeddings",
    make_repo=False,
)

recRAG.save_user_embeddings("SeppeV/user_embeddings_test")
