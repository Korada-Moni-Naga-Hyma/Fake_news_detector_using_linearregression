import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')

df=pd.read_csv('train.csv')
df=df.fillna(' ')
df['content']=df['author']+" "+df['title']
X=df.drop('label',axis=1)
Y=df['label']

ps=PorterStemmer()
def stemming(content):
  stemmed_content=re.sub('[^a-zA-Z]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content=[ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content=' '.join(stemmed_content)
  return stemmed_content

df['content'] = df['content'].apply(stemming)

X=df['content'].values
Y=df['label'].values
vector=TfidfVectorizer()
vector.fit(X)
X=vector.transform(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=1)
model=LogisticRegression()
model.fit(X_train,Y_train)

#website
st.title('Fake news Detector')
input_text=st.text_input('Enter news article')

def prediction(input_text):
    try:
        # Wrap input_text in a list
        input_data = vector.transform([input_text])
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error("Error during prediction: {e}")
        return None


if input_text:
  pred=prediction(input_text)
  if pred==1:
    st.write('The News is Fake')
  else:
    st.write('The News is Real')
  