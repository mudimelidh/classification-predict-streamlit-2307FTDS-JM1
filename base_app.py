"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
def header(title, subheader):
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	image = Image.open("resources/imgs/global.jpg")
	st.image(image, caption = "Global warming")
	return 

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
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
	
	# Building out the statistics page
	if selection == "Statistics":
		st.title("Statistics")
		st.subheader("Summary of Statistics")
		st.info("Summary statistics")
		st.bar_chart(raw["sentiment"])
        
    #if selection == "Statistics":
     #   st.title("Statistics")
     #   st.subheader("Summary of Statistics")
     #   st.info("Summary statistics")

    # Display Pie Chart
    st.subheader("Sentiment Distribution (Pie Chart)")

    # Create a pie chart
    sentiment_counts = raw["sentiment"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart using st.pyplot
    st.pyplot(fig)
        
     
   
    

            
		# Build it here add more stuff
	
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		models = ["Logistic Regressor", "Random Forest", "Neural Network"]
		model_selected = st.selectbox("Select Prediction Model", models)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if model_selected == "Logistic Regressor":
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_selected == "Random Forest":
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			else:
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			if prediction == 2:
				prediction = "News: the tweet links to factual news about climate change"
			elif prediction == 1:
				prediction = "Pro: the tweet supports the belief of man-made climate change"
			elif prediction == 0:
				prediction = "Neutral: the tweet neither supports nor refutes the belief of man-made climate change"
			else:
				prediction = "*Anti*: the tweet does not believe in man-made climate change Variable definitions"



			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
