
import streamlit as st
import sys
import pathlib

sys.path.append(str(pathlib.Path().absolute()).split("/src")[0])

print(sys.path)
import datetime
import subprocess
import torch
import numpy as np
import pandas as pd
import whisper


from haystack.retriever.dense import EmbeddingRetriever
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

@st.cache(allow_output_mutation=True)
def load_model():
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
    return pipe
pipe = load_model()



st.set_page_config(page_title="AI talk search", page_icon="🤖", layout="centered")

st.title("Search AI talks")

st.sidebar.write(
    f"Search AI talks from prominent researchers"
)

st.sidebar.write(
    f"Add videos to search [....]"
)

form = st.form(key="annotation")

with form:
    cols = st.columns((1,1))
    query = cols[0].text_input("Query transcripts")
    entity = cols[1].selectbox(
        "Entity:", ["Andrej Kaparthy", "Yann Lecun", "Fei Fei"], index=2
    )
  
    # slider_ex = cols[1].slider(" param", 1, 5, 2)
    submitted = st.form_submit_button(label="ASK")

text="NO"
if submitted:
    text="YES"
    
    prediction = pipe.run(query="What biotech events are happening in San Francisco ?", params={"Retriever": {"top_k": 40}, "Reader": {"top_k": 25}})
    
    st.success("Results.")
    st.balloons()
    st.write(f"{text}")
    st.write(prediction)
    # st.write(f"{prediction}")


# expander = st.expander("See all records")
# with expander:
    # st.dataframe(get_data(gsheet_connector))