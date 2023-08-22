import streamlit as st
import pandas as pd
import cv2
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import neattext.functions as nfx


def main():
    def sidebar(logo_path ):
        logo_image = logo_path

        # Add a title and header in the sidebar
        st.sidebar.title("MATRIC NUMBERS :: CPE/2017/1031 \t  CPE/2017/1019")

        # Display the logo
        st.sidebar.image(logo_image, width=100)

        # Display the additional content
        st.sidebar.markdown("## App Information")
        st.sidebar.write("Welcome to the Emotion App!")
        st.sidebar.write("This app predicts emotions based on images and text.")
        st.sidebar.write("Upload an image and enter text, then click 'Predict' to see results.")
    sidebar("logo.png")


    # Load the trained model
    model = load_model('combine1_model_25.h5')

    # Define the LSTM input shape

    max_sequence_length = 31
    st.title("Emotion Prediction App")
    # Get user inputs
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])
    text_input = st.text_input("Enter text")
    predict_button = st.button("Predict")

    text_data = {"text": [text_input]}
    text_data = pd.DataFrame(text_data)

    def preprocess(df, text_column):
        # Check if the specified text_column exists in the DataFrame
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the DataFrame.")

        # Clean the text column
        df[text_column] = df[text_column].str.lower()
        df[text_column] = df[text_column].apply(nfx.remove_userhandles)
        df[text_column] = df[text_column].apply(nfx.remove_punctuations)
        df[text_column] = df[text_column].apply(nfx.remove_stopwords)
        df[text_column] = df[text_column].apply(nfx.remove_phone_numbers)
        df[text_column] = df[text_column].apply(nfx.remove_html_tags)
        df[text_column] = df[text_column].apply(nfx.remove_special_characters)
        df[text_column] = df[text_column].apply(nfx.remove_numbers)

        return df

    df = preprocess(text_data, 'text')

    # Text Data Processing

    def preprocess_text(text, max_sequence_length):
        df1 = pd.read_csv('Emotion.csv')
        df1['Clean_Text'] = df1['Clean_Text'].astype(str)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df1["Clean_Text"])
        sequences = tokenizer.texts_to_sequences(df["text"])
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
        vocabulary_size = len(tokenizer.word_index) + 1
        return padded_sequences, vocabulary_size

    # Image Data Processing
    def preprocess_image(image_content, target_size=(48, 48)):
        image = np.asarray(bytearray(image_content.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, target_size)
        image = image.astype('float32') / 255.0
        image = image.reshape(1, *target_size, 1)
        return image

    class_names = ['depression', 'happy', 'surprise']

    if predict_button and uploaded_image is not None and text_input:

        # Preprocess the text data and image
        processed_image = preprocess_image(uploaded_image)
        processed_text, vocabulary_size = preprocess_text(df['text'], 31)

        # Make predictions
        predictions = model.predict([processed_image, processed_text])

        # Get the probabilities for each class from the output arrays
        cnn_probabilities = predictions[0][0]
        lstm_probabilities = predictions[1][0]

        # Display the results
        st.write("CNN Probabilities:")
        for class_name, probability in zip(class_names, cnn_probabilities):
            st.write(f"{class_name}: {probability * 100:.2f}%")

        st.write("\n")

        st.write("LSTM Probabilities:")
        for class_name, probability in zip(class_names, lstm_probabilities):
            st.write(f"{class_name}: {probability * 100:.2f}%")


if __name__ == '__main__':
    main()
