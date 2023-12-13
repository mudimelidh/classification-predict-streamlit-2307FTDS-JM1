"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: 2307FTDS Team JM1.

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
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import re
import string
from io import BytesIO


# Data dependencies
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

st.markdown(
	"""
	<style>
	.main{
	background-color: #C1E1C1;
	color: #555555
	}
	h1, h2, h3 {
    color: #4D4D4D;
    }
	p, ol, ul, dl {
	color: #4D4D4D
	}
	</style>
	""",
	unsafe_allow_html=True
)

# Vectorizer
message_vectorizer = open("resources/tfidf_vectorizer.pkl","rb")
tweet_vc = joblib.load(message_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")
def header(title, subheader):
	logo_col, title_col = st.columns(2)
	image = Image.open("resources/imgs/logot.png")
	logo_col.image(image)
	title_col.title(title)
	st.markdown('<hr style="border: 2px solid #228B22;">', unsafe_allow_html=True)
	st.subheader(subheader)

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

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creating sidebar with selection box
	options = ["Predictor", "Tweeter Data", "Statistics", "Meet the Team"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Tweeter Data":
		header("Tweeter Data", "Tweet train Data")
		info_col, logo_col = st.columns(2)
		info_col.info("Welcome to the Twitter Data Page, your gateway to a wealth of insights derived from the pulse of social discourse. Here, we present a curated collection of tweets meticulously analyzed through advanced machine learning algorithms.")
		image = Image.open("resources/imgs/tweet.png")
		logo_col.image(image, width=50)
		# You can read a markdown file from supporting resources folder

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
	
	# Building out the statistics page
	if selection == "Statistics":
		header("Sentiment Statistics", "📊 Welcome to the Statistics Page")
		st.info("Your hub for insightful metrics and analytics on our Twitter Classifier for Man-Made Global Warming")
		
		st.subheader("Sentiment Distribution (Pie Chart)")
        # Create a pie chart

		# Create a DataFrame or use your existing 'raw' DataFrame
		sentiment_counts = raw["sentiment"].value_counts()
		# Adjust the size and background transparency of the pie chart
		fig, ax = plt.subplots()  # Set the size as needed
		categories = ["Pro","News","Neutral","Anti"]
		ax.pie(sentiment_counts, labels=categories, autopct='%1.1f%%', startangle=90)
		ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

		# Make the background transparent
		ax.patch.set_alpha(0)

		# Save the figure to a BytesIO object
		image_stream = BytesIO()
		plt.savefig(image_stream, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
		plt.close()

		# Display the pie chart using st.image
		st.image(image_stream)

        # Display Word Clouds
		st.subheader("Word Clouds for Each Sentiment Class")
		# Add interactive selection buttons for word clouds
		selected_sentiment = st.selectbox("Select Sentiment Class", ["Anti","Pro","News","Neutral"])
		if selected_sentiment == "Anti":
			selected_sentiment = -1
		elif selected_sentiment == "Pro":
			selected_sentiment = 1
		elif selected_sentiment == "News":
			selected_sentiment = 2
		else:
			selected_sentiment = 0
		generate_word_cloud(selected_sentiment, raw)
	
	# Building out the predication page
	if selection == "Predictor":
		header("Welcome to the Tweet Classifier!", "🚀 Explore Perspectives on Climate Change")
		st.info(
			"""
			Select from a range of powerful machine learning models designed to analyze tweets and
			  uncover sentiments regarding man-made global warming. Whether you're curious about public opinion,
			    tracking trends, or researching climate discourse, our Tweet Classifier is here to assist.
			"""
			)
		models = ["Support Vector Classifier", "Logistic Regressor", "Random Forest", "Decision Tree"]
		model_selected = st.selectbox("Select Prediction Model", models)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			# vect_text = tweet_cv.transform([tweet_text]).toarray()
			X = tweet_vc.transform([tweet_text]).toarray()

			if model_selected == "Logistic Regressor":
				predictor = joblib.load(open(os.path.join("resources/lr_model.pkl"),"rb"))
				prediction = predictor.predict(X)
			elif model_selected == "Random Forest":
				predictor = joblib.load(open(os.path.join("resources/rf_model.pkl"),"rb"))
				prediction = predictor.predict(X)
			elif model_selected == "Support Vector Classifier":
				predictor = joblib.load(open(os.path.join("resources/svc_model.pkl"),"rb"))
				prediction = predictor.predict(X)
			else:
				predictor = joblib.load(open(os.path.join("resources/decision_tree_model.pkl"),"rb"))
				prediction = predictor.predict(X)

			if prediction == 2:
				prediction = "**News**: the tweet links to factual news about climate change"
			elif prediction == 1:
				prediction = "**Pro**: the tweet supports the belief of man-made climate change"
			elif prediction == 0:
				prediction = "**Neutral**: the tweet neither supports nor refutes the belief of man-made climate change"
			else:
				prediction = "**Anti**: the tweet does not believe in man-made climate change"

			# When model has successfully run, will print prediction
			st.success("The tweet is Categorized as: {}".format(prediction))
	
	if selection == "Meet the Team":
		header("Meet our Team", "")

		image_path = "resources/imgs/dakalo.png" 
		# Center the image using custom CSS
		st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" alt="Dakalo Profile Image" style="width: 150px; max-width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    	)
		st.markdown('<div style="text-align: center;"><strong>🚀 Dakalo Mudimeli: Team Leader & Data Scientist</strong></div>', unsafe_allow_html=True)
		st.info("Greetings! I'm Dakalo Mudimeli, an accomplished Aeronautical Engineering graduate who found a passion for harnessing data to navigate the complex challenges of our world. As the Team Leader and a Data Scientist on our innovative project, I lead a talented group of data scientists in a mission to classify tweets and predict individuals\' beliefs in man-made global warming.")
		st.markdown('<hr style="border: 2px solid #228B22;">', unsafe_allow_html=True)
		
		one, two = st.columns(2)

		# Percy Project manager
		image_path = "resources/imgs/percy.png"
		one.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" alt="Percy Image" style="width: 150px; max-width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    	)
		one.markdown('<div style="text-align: center;"><strong>🌟 Percy Mmutle: Project Manager & Data Scientist</strong></div>', unsafe_allow_html=True)
		one.info("Greetings! I'm Percy Mmutle, Skilled in orchestrating machine learning projects, ensuring seamless collaboration, and achieving project milestones.")
		
		# Ntombenhle Researcher
		image_path = "resources/imgs/Ntombenhle.png"
		two.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" alt="Ntomenhle Image" style="width: 150px; max-width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    	)
		two.markdown('<div style="text-align: center;"><strong>🌟 Ntombenhle Nkosi: Project Researcher & Data Scientist</strong></div>', unsafe_allow_html=True)
		two.info("Greetings! I'm Ntombenhle, a chemical engineer turned into a data analyst enthusiast. As the Project Researcher of our innovative project, I bring my specialized research strengths to the forefront, focusing on harnessing the potential of machine learning to predict public beliefs on man-made global warming.")
		#
		st.markdown('<hr style="border: 2px solid #228B22;">', unsafe_allow_html=True)
		
		# Another section for the other 3 team members with three columns
		one, two, three = st.columns(3)

		# Nyeleti group scrum secretary
		image_path = "resources/imgs/nyeleti.png"
		one.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" alt="Nyeleti Image" style="width: 150px; max-width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    	)
		one.markdown('<div style="text-align: center;"><strong>🤖 Nyeleti Chauke: Administrator & Data Scientist</strong></div>', unsafe_allow_html=True)
		one.info("Greetings! I'm Nyeleti Chauke, Administrator for DataNova. I am responsible for making sure that everyone is aware of their duties and responsibilities, and to ensure that tasks are executed. ")

		# Sharonrose Project administrator
		image_path = "resources/imgs/sharonrose.png"
		two.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" alt="Sharon Image" style="width: 150px; max-width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    	)
		two.markdown('<div style="text-align: center;"><strong>🌟 Sharonrose Khokhololo: Project Admin & Data Scientist</strong></div>', unsafe_allow_html=True)
		two.info("Greetings! I'm Sharonrose Khokhololo, I ensure smooth coordination, clear communication, and efficient project execution to contribute to the success of our mission.")
		
		# Khaya GitHub reviewer
		image_path = "resources/imgs/profile.png"
		three.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" alt="Khaya Image" style="width: 150px; max-width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    	)
		three.markdown('<div style="text-align: center;"><strong>🚀 Khaya Bresendale-Fynn: GitHub Reviewer & Data Scientist</strong></div>', unsafe_allow_html=True)
		three.info("Greetings! I'm Khaya Bresendale-Fynn, I am the Github reviewer for the project.")


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
