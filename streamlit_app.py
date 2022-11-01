
import streamlit as st

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
    author = cols[0].text_input("Query transcripts")
    bug_type = cols[1].selectbox(
        "Bug type:", ["Andrej Kaparthy", "Yann Lecun", "Fei Fei"], index=2
    )
  
    # slider_ex = cols[1].slider(" param", 1, 5, 2)
    submitted = st.form_submit_button(label="Submit")

text="NO"
if submitted:
    text="YES"
    st.success("Results.")
    st.balloons()
    st.write(f"{text}")


# expander = st.expander("See all records")
# with expander:
    # st.dataframe(get_data(gsheet_connector))