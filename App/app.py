# Core Packages
import altair as alt
import streamlit as st

# EDA Packages
import pandas as pd
import numpy as np

# Utils
import joblib

# Functions
pipe_lr = joblib.load(open("models/emotion_classifer_pipe_lr_27_July_2022.pkl", "rb"))

def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
    st.title("Emotion Classifer App")
    menu = ['Home','Monitor','About']

    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home-Emotion In Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2= st.columns(2)

            # Calling the functions here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                probability_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                # st.write(probability_df.T)
                probability_df_clean = probability_df.T.reset_index()
                probability_df_clean.columns = ["emotions", "probability"]

                dig = alt.Chart(probability_df_clean).mark_bar().encode(x='emotions', y='probability',color='emotions')
                st.altair_chart(dig, use_container_width=True)


    elif choice == "Monitor":
        st.subheader("Monitor App")

    else:
        st.subheader("About")


if __name__ == '__main__':
    main()