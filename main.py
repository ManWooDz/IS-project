import streamlit as st

pages = {
    "Documents": [
        st.Page("pages/ML-docs.py", title="Machine Learning Documentation"),
        # st.Page("pages/", title="Neural Language Documentation"),
    ],
    "Models": [
        st.Page("pages/ML-model.py", title="Machine Learning Models"),
        # st.Page("pages/", title="Neural Language Models"),"),
    ],
}

pg = st.navigation(pages)
pg.run()