# Streamlit dependencies
import streamlit as st
import joblib
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import os
import re
import string

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    text = re.sub(r'\brt\b', '', text)

    # Remove URLs using regular expression
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text

def generate_word_cloud(sentiment_class, df_train):
    sentiment_data = df_train[df_train['sentiment'] == sentiment_class]['message']
    
    # Preprocess the text
    sentiment_data = sentiment_data.apply(preprocess_text)
    
    sentiment_text = ' '.join(sentiment_data)

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)

    # Display the word cloud using st.image
    st.image(wordcloud.to_image(), caption=f'Word Cloud for {sentiment_class} Sentiment', use_column_width=True)



def header(title, subheader):
    st.title("Tweet Classifier")
    st.subheader("Climate change tweet classification")
    image = Image.open("resources/imgs/global.jpg")
    st.image(image, caption="Global warming")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "Statistics"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']])  # will write the df to the page

    # Building out the statistics page
    elif selection == "Statistics":
        st.title("Statistics")
        st.subheader("Summary of Statistics")
        st.info("Summary statistics")

        # Display Pie Chart
        st.subheader("Sentiment Distribution (Pie Chart)")

        # Create a pie chart
        sentiment_counts = raw["sentiment"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Display the pie chart using st.pyplot
        st.pyplot(fig)

        # Display Word Clouds
        st.subheader("Word Clouds for Each Sentiment Class")

        # Add interactive selection buttons for word clouds
        selected_sentiment = st.selectbox("Select Sentiment Class", raw['sentiment'].unique())
        generate_word_cloud(selected_sentiment, raw)

    # Building out the prediction page
    elif selection == "Prediction":
        st.info("Prediction with ML Models")
        models = ["Logistic Regressor", "Random Forest", "Neural Network"]
        model_selected = st.selectbox("Select Prediction Model", models)
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()

            # Load the corresponding model file for the selected model
            if model_selected == "Logistic Regressor":
                predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
            elif model_selected == "Random Forest":
                predictor = joblib.load(open(os.path.join("resources/Random_forest.pkl"), "rb"))
            else:
                predictor = joblib.load(open(os.path.join("resources/Neural_network.pkl"), "rb"))

            prediction = predictor.predict(vect_text)

            if prediction == 2:
                prediction = "News: the tweet links to factual news about climate change"
            elif prediction == 1:
                prediction = "Pro: the tweet supports the belief of man-made climate change"
            elif prediction == 0:
                prediction = "Neutral: the tweet neither supports nor refutes the belief of man-made climate change"
            else:
                prediction = "*Anti*: the tweet does not believe in man-made climate change Variable definitions"

            # When the model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
