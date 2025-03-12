import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model

word_index= imdb.get_word_index()
reverse_word_index= { value:key for key,value in word_index.items()}

model= load_model('simplernn.h5')

#? function to decode review
def decode_review(encoded_review):
    return ' '.join(rev_word_index.get(i-3,'?')for i in encoded_review)
#? function for preprocessing
def preprocessing_text(text):
    words= text.lower().split()
    encoded_review= [word_index.get(word,2)+3 for word in words]
    padded_review =sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


#prediction function
### prediction function
def predict_sentiment(review):
    preprocessed_review=preprocessing_text(review)
    prediction= model.predict(preprocessed_review)
    sentiment ='negative'
    if prediction[0][0]>0.5:
        sentiment ='positive'
    return sentiment,prediction[0][0]

import streamlit as st
st.title('IMDB movie review sentiment analysis using simple RNN')
st.write('Enter moview name')

user_input=st.text_area('movie review')

if st.button('Classify'):
    
    sentiment,score=predict_sentiment(user_input)
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction score:{score}')
else:
    st.write('please enter the movie review')
    