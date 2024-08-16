import tensorflow as tf
import streamlit as st 
from tensorflow.keras.utils import pad_sequences
import pickle

model = tf.keras.models.load_model('Next_word_pred.keras')
tokenizer = pickle.load(open('Next_word_tokenizer.pkl', 'rb'))

def next_word_pred(text, model, tokenizer, maxlen=338):
    word_index = tokenizer.word_index
    input_token = tokenizer.texts_to_sequences([text])[0]
    padded_input = pad_sequences([input_token], maxlen=maxlen)
    preds = model.predict(padded_input).argmax()
    for word, index in word_index.items():
        if index == preds:
            return word, index
    
# text = 'children were playing'
# print(next_word_pred(text, model, tokenizer))

st.title('Next word Prediction using LSTM')
input_text = st.text_area("Enter the text: ")

try:
    if st.button('Click to predict'):
        st.write("Predicted: ")
        st.write(next_word_pred(input_text, model, tokenizer)[0])
except Exception as e:
    print(e)