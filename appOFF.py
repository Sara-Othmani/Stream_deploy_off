#all the imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import nltk
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import streamlit as st
from PIL import Image 
import pip

#loading the dataset
df_offres = pd.read_excel('./OFFRES.xlsx')

#changing a wrong value
df_offres.at[1, 'Nom_entreprise'] = 'McKinsey & Company'

#cleaning and pre-processing the column of features
#downloadig the stopwords, punkt and wordnet using the NLTK downloader
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#creating a new column for all the features/criteria
df_offres['features_off'] = df_offres['Nom_offre']+" "+df_offres['Secteur_act_job']+" "+df_offres['Diplome']+" "+df_offres['Experience_requise'].map(str)+" "+df_offres['Qualifications']

from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

def black_txt(token):
    return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2   
  
def clean_txt(text):
    clean_text = []
    clean_text2 = []
    text = re.sub("'", "",text)
    text=re.sub("(\\d|\\W)+"," ",text) 
    text = text.replace("nbsp", "")
    clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)

#creating a sub dataset on which I will perform the content-based algorithm
offres = df_offres[['id_offre','Nom_offre', 'Nom_entreprise', 'features_off']]

#applying the cleaning part on the features column
offres.loc[:, 'features_off'] = offres['features_off'].apply(clean_txt)

cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(offres['features_off']).toarray()

similarity = cosine_similarity(vector)

def recommend(off):
    index = offres[offres['Nom_offre'] == off].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    recommendations = []
    for i in distances[1:4]:
        offre_name = offres.iloc[i[0]]['Nom_offre']
        entreprise_name = offres.iloc[i[0]]['Nom_entreprise']
        recommendations.append((offre_name, entreprise_name))
    return recommendations

# Create the Streamlit app
def app():
    st.title("Job Offer Recommender")
    # Get user input
    job_position = st.text_input("Enter a job position")
    # Show recommendations
    if job_position:
        recommendations = recommend(job_position)
        if recommendations:
            st.write("Here are the top job offers for the position of", job_position)
            for recommendation in recommendations:
                st.write("Offer Name:", recommendation[0])
                st.write("Company name:", recommendation[1])
        else:
            st.error("No job offers available are found for the position.")

if __name__ == '__main__':
    app()