from cgitb import text
import streamlit as st
from SA import Sentiment_Analysis

sa = Sentiment_Analysis()

def processtweet(tweet):
    preprocessed_text = sa.cleanTweet(tweet)
    tfidfVector = sa.convertTFIDF(preprocessed_text)
    tfidfArray = tfidfVector.toarray()
    weights = sa.loadWeights()
    
    return tfidfArray, weights

def updatePos(tfidfArray, prediction):
    if prediction == 'Positive':
        label = 1
    elif prediction == 'Negative':
        label = 0
    sa.updateTfidf(tfidfArray, label)
    print("UpdatePos")
    

def updateNeg(tfidfArray, prediction):
    if prediction == 'Positive':
        label = 0
    elif prediction == 'Negative':
        label = 1
    sa.updateTfidf(tfidfArray, label)
    print("UpdateNeg")


if 'previous_inputs' not in st.session_state:
    st.session_state.previous_inputs = []

# Create a text input for the user
tweet = st.text_input("Enter your input:")

if tweet in st.session_state.previous_inputs:
    st.error("You've already entered this input before. Please try again!")
else:
    # Create a button to trigger the prediction
    if st.button("Get Prediction"):
        st.session_state.previous_inputs.append(tweet)
        # Generate a prediction (replace with your own prediction logic)
        if len(tweet) > 0:
            tfidfArray, weights = processtweet(tweet)
            prediction = sa.predict(tfidfArray, weights)
            st.write(f"Prediction: {prediction}")
            
            if "counter" not in st.session_state:
                st.session_state.counter = 0
            
            st.session_state.counter += 1
            # update on button changes
            print(st.session_state.counter, " : ", tweet)
            st.write("Is the Prediction correct: ")

            # Create a feedback section with thumbs up and thumbs down buttons
            col1, col2 = st.columns(2)
            with col1:
                st.button(label="👍", use_container_width=True, on_click=updatePos, args=(tfidfArray, prediction))
            with col2:
                st.button(label="👎", use_container_width=True, on_click=updateNeg, args=(tfidfArray, prediction))
    else:
        st.warning("No new tweet entered")