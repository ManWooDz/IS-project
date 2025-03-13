import streamlit as st

pages = {
    "Documents": [
        st.Page("pages/ML-docs.py", title="Machine Learning Documentation"),
        # st.Page("pages/", title="Neural Network Documentation"),
    ],
    "Models": [
        st.Page("pages/ML-model.py", title="Machine Learning Models"),
        st.Page("pages/NN-predict.py", title="Neural Network Models"),
        # st.Page("pages/NN-model.py", title="Neural Network Models"),
    ],
}

pg = st.navigation(pages)
pg.run()