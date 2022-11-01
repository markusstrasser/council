# Expert Council Bot

[Link to Live Web App](INSERT LINK HERE)

## Project Motivations

We want to know what Person _x_ would say about _y_ or how they might answer question _z_

## Project Description

### Technologies Used

[Pytube](https://pytube.io/en/latest/) - Audio file extraction from YouTube playlists of experts talking about a subject (current MVP is specific to topics of AI)<br>
[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) - Transcription of audio<br>
Haystack [Pipelines API](https://docs.haystack.deepset.ai/reference/pipelines-api) and [ReaderAPI](https://docs.haystack.deepset.ai/reference/reader-api) - Synthesis for modeling a representation of an expert<br>
[Streamlit](https://streamlit.io/) - front end web app deployment

## User Flow

1. Prompt user
> "What is self-supervised learning" 
2. The model would find what the expert would reply to this. 
4. The model returns what it finds as the best answer, and the original source of the data.  

## Constructing an "Expert" Council Member
1. Get raw audio content of a monologue of our target person speaking, correctly labeled. 
2. Label the objects in a dictionary with that speaker. 
3. Pass the audio to extract a whisper object
4. Populate a dictionary with metadata, the transcript file, and the audio/link to source of data



## Features:

**1. Return the Most Relevant Content from an Expert on Prompt's Subject Matter**

Match our database to the prompt inputs as closely as possible. return the object in our dictionary with the person, the material or content of what they mention, and relevant meta data such as the date they said it and link to the original source. 

**2. Evolution on prompt's subject matter** 
Metadata from YouTube link (original audio source) to get the upload date. 

If "neural network" comes up, we know when the expert's stance or mentioning of this 


## What Could Be Done In The Future 

1. Expand database to more content from multiple platforms and appearances of each expert, and adding more experts. 

This requires **speaker segmentation** on the raw audio file, labeling the expert, and capturing the relevant meta data from the audio source. 
Building out a pipeline from other audio/video sources than just YouTube monologues.

2. Database building feature to add to the model: Passing in audio start and end times. 
i. Pass in audio start and end time along with URL
ii. Trim audio from URL to start and end times


## Libraries Used
[pytube](https://pytube.io/en/latest/index.html)

[whisper](https://github.com/openai/whisper)

