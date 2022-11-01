import whisper
import datetime
import subprocess
import torch
import numpy as np
import pandas as pd

from haystack.retriever.dense import EmbeddingRetriever
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

#write a function that adds two numbers

df = pd.read_json("example_whisper_transcript")
print(df)
ds = df.to_dict("records")

document_store = InMemoryDocumentStore(embedding_dim=384, similarity="cosine")

# document_store = ElasticsearchDocumentStore(similarity="cosine",
#                                             embedding_dim=384)

dicts = [{"content": d["text"], "meta": {i:d[i] for i in d if i!='text'}} for d in ds]

document_store.write_documents(dicts)
retriever = EmbeddingRetriever(document_store=document_store, embedding_model='sentence-transformers/all-MiniLM-L6-v2', use_gpu=True, top_k=10)

document_store.update_embeddings(retriever)
reader = FARMReader(model_name_or_path='deepset/roberta-base-squad2', use_gpu=True)
# result = reader.predict(
#     query="Which country is Canberra located in?",
#     documents=documents,
#     top_k=10
# )
pipe = ExtractiveQAPipeline(reader, retriever)
# You can configure how many candidates the reader and retriever shall return
# The higher top_k_retriever, the better (but also the slower) your answers. 
prediction = pipe.run(
    query="What biotech events are happening in San Francisco ?", params={"Retriever": {"top_k": 40}, "Reader": {"top_k": 25}}
)



