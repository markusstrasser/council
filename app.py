
import streamlit as st
import sys
import pathlib

# sys.path.append(str(pathlib.Path().absolute()).split("/src")[0] ")

# print(sys.path)
import datetime
import subprocess
import torch
import numpy as np
import pandas as pd
import whisper
import json

from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore 
from haystack.pipelines import ExtractiveQAPipeline


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


@st.cache(allow_output_mutation=True)
def load_model():
    #write a function that adds two numbers

    # create an initial empty pd dataframe to store the data
    df = pd.DataFrame()


    ds = []
    for transcript in pathlib.Path("./whisper_objects/").glob("*.json"):
        with open(transcript) as f:
            data = json.loads(f.read())
            for d in data:
                if type(d) == dict:
                    ds += [{"text": d["text"]}]

    # transcripts = "example_whisper_transcript.json"
    # with open(transcripts) as f:
    #     data = json.loads(f.read())
    # #print(data['data'][0])
    # df = pd.DataFrame.from_records(data["segments"])
    # print(df)

    # ds = df.to_dict("records")


    #! THIS CAUSES SEGMENTATION FAULTS? 
    document_store = InMemoryDocumentStore(embedding_dim=768, similarity="cosine")

    # document_store = ElasticsearchDocumentStore(similarity="cosine",
    #                                             embedding_dim=384)
    # dicts = [{"content": d["text"], "meta": {i:d[i] for i in d if i!='text'}} for d in ds]
    dicts = [{"content": d["text"]} for d in ds]

    document_store.write_documents(dicts[:10])
    print("doc store")
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model='sentence-transformers/all-mpnet-base-v2', use_gpu=True, top_k=10)

    print("retriever")
    document_store.update_embeddings(retriever)
    reader = FARMReader(model_name_or_path='deepset/roberta-base-squad2', use_gpu=True)
    # result = reader.predict(
    #     query="Which country is Canberra located in?",
    #     documents=documents,
    #     top_k=10
    # )
    print("before pipe")
    pipe = ExtractiveQAPipeline(reader, retriever)
    return pipe

pipe = load_model()


if submitted:
    text="YES"
    
    prediction = pipe.run(query=query, params={"Retriever": {"top_k": 40}, "Reader": {"top_k": 25}}) #maybe reduce these numbers so they're faster 
    
    st.success("Results.")
    st.balloons()
    st.write(f"{text}")
    print(type(prediction), prediction)
    answers = [a for a in prediction["answers"]]
    print("\n \n ~~~ ANSWERS ~~~ \n \n", type(answers[0]), answers[0])
    print("\n answers[0][answer] \n ", answers[0]["answer"])
    st.write([a for a in prediction["answers"]])
    # st.write(f"{prediction}")



# expander = st.expander("See all records")
# with expander:
    # st.dataframe(get_data(gsheet_connector))