import streamlit as st
import pickle as pkl
import string
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download()
ps = PorterStemmer()

def trans_text(txt):
    txt = txt.lower()
    txt = nltk.word_tokenize(txt)
    y=[]
    for i in txt:
        if i.isalnum():
            y.append(i)
    txt = y[:]
    y.clear()
    for i in txt:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    txt = y[:]
    y.clear()
    for i in txt:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pkl.load(open('vectorizer.pkl','rb'))
model = pkl.load(open('model.pkl','rb'))

st.title("SMS/Email Spam Classifier")
input_sms = st.text_input("Enter the Message")
if st.button('Predict'):
    # 1.preprocess
    trans_sms = trans_text(input_sms)
    # 2.vectorize
    vector_input = tfidf.transform([trans_sms])
    # 3.predict
    result = model.predict(vector_input)[0]
    # 4.result
    if(result==1):
        st.header("Spam")
    else: 
        st.header("Not Spam")

