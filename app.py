#Loading EFA Packages
import pandas as pd
import numpy as np

#Load Data Viz  Packages
import seaborn as sns
import matplotlib.pyplot as plt

#Load Text Cleaning Packages
import neattext.functions as nfx

#Load ML Packages
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Load Dataset
df = pd.read_csv("data/emotion_dataset_2.csv")

# Value Counts
(df['Emotion'].value_counts())

sns.countplot(x='Emotion', data=df)
#plt.show()

#Data Cleaning
dir(nfx)

# User handles
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

# Stopwords
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

# Stopwords
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_special_characters)

# Features & Labels
xfeatures = df['Clean_Text']
ylabels = df['Emotion']

# Split Data
x_train, x_test, y_train, y_test = train_test_split(xfeatures,ylabels,test_size=0.3,random_state=42)

# Build Pipline
from sklearn.pipeline import Pipeline

# LogisticRegression Pipeline
pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression())])

# Train and Fit Data
pipe_lr.fit(x_train, y_train)

# Check Accuracy
pipe_lr.score(x_test,y_test)

# Make a prediction
ex1 = "Wow Dion's penis is very small!"

print(pipe_lr.predict([ex1]))

# Prediction Prob
print(pipe_lr.predict_proba([ex1]))

# To Know the classes
print("-----CLASSES------")
print(pipe_lr.classes_)


# Saving the model and pipeline
import joblib
pipeline_file = open("models/emotion_classifer_pipe_lr_27_July_2022.pkl", "wb")
joblib.dump(pipe_lr,pipeline_file)
pipeline_file.close()
