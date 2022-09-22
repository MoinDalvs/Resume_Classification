import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import click
import spacy
import docx2txt
import pdfplumber
from pickle import load
import requests
import re
import os
import PyPDF2
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
# load pre-trained model
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot  as plt
stop=set(stopwords.words('english'))
import aspose.words as aw
from spacy.matcher import Matcher
# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)


mfile = BytesIO(requests.get('https://github.com/MoinDalvs/Resume_Classification/blob/main/model.sav?raw=true').content)
model = load(mfile)
mfile1 = BytesIO(requests.get('https://github.com/MoinDalvs/Resume_Classification/blob/main/model_id.pkl?raw=true').content)
model1 = load(mfile1)

def add_bg_image():
    st.markdown(
          f"""
          <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-vector/abstract-background-with-squares_23-2148995948.jpg?w=996&t=st=1663219978~exp=1663220578~hmac=aee3da925492e169a7f9fb7d1aa1577c58a7db3849d8be3f448114080d42a7a7");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True)
     
add_bg_image()

def extract_skills(resume_text):

    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
            
    # reading the csv file
    data = pd.read_csv("https://raw.githubusercontent.com/MoinDalvs/Resume_Classification/main/skills.csv") 
            
    # extract values
    skills = list(data.columns.values)
            
    skillset = []
            
    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)
            
    # check for bi-grams and tri-grams (example: machine learning)
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
            
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

# Function to extract text from resume
def getText(filename):
      
    # Create empty string 
    fullText = ''
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        
        for para in doc:
            fullText = fullText + para
            
           
    else:  
        with pdfplumber.open(filename) as pdf_file:
            pdoc = PyPDF2.PdfFileReader(filename)
            number_of_pages = pdoc.getNumPages()
            page = pdoc.pages[0]
            page_content = page.extractText()
             
        for paragraph in page_content:
            fullText =  fullText + paragraph
         
    return (fullText)

def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))

    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
            
    return resume

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)  


# Function to extract experience details
def expDetails(Text):
    global sent
   
    Text = Text.split()
   
    for i in range(len(Text)-2):
        Text[i].lower()
        
        if Text[i] ==  'years':
            sent =  Text[i-2] + ' ' + Text[i-1] +' ' + Text[i] +' '+ Text[i+1] +' ' + Text[i+2]
            l = re.findall(r'\d*\.?\d+',sent)
            for i in l:
                a = float(i)
            return(a)
            return (sent)

target = {0:'Peoplesoft',1:'SQL Developer',2:'React JS Developer',3:'Workday'}

def main():
    html_temp = """
    <div style ="background-color:transparent;padding:13px">
    <h1 style ="color:black;text-align:center;"> RESUME CLASSIFICATION </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    file_type=pd.DataFrame([], columns=['Uploaded File', 'Experience', 'Skills', 'Predicted Profile'])
    filename = []
    predicted = []
    experience = []
    skills = []
    
    upload_file = st.file_uploader('Hey,Upload Your Resume ',
                                type= ['docx','pdf'],accept_multiple_files=True)

    for doc_file in upload_file:
            if doc_file is not None:
                filename.append(doc_file.name)
                cleaned=preprocess(display(doc_file))
                prediction = model.predict(model1.transform([cleaned]))[0]
                predicted.append(target.get(prediction))
                extText = getText(doc_file)
                exp = expDetails(extText)
                experience.append(exp)
                skills.append(extract_skills(extText))
    if len(predicted) > 0:
        file_type['Uploaded File'] = filename
        file_type['Experience'] = experience
        file_type['Skills'] = skills
        file_type['Predicted Profile'] = predicted
        # file_type
        # Custom formatting
        st.table(file_type.style.format({'Experience': '{:.1f}'}))

    st.subheader("About") 
    st.info("This project is a part of AiVariant Internship")
    st.markdown('<img align="center" alt="code"  height="180" width="720" src = "https://user-images.githubusercontent.com/99672298/186088252-ef41cefc-cda7-49e8-960d-efbd59d3696c.jpg">\
\
<table>\
<tr>\
<td>\
     <img src="https://avatars.githubusercontent.com/u/99672298?v=4" width="180"/>\
     \
     moindalvs@gmail.com\
\
<p align="center">\
<a href = "https://github.com/MoinDalvs"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>\
<a href = "https://twitter.com/DalvsHubot"><img src = "https://www.shareicon.net/download/2016/07/06/107115_media.svg" width="36" height="36"/></a>\
<a href = "https://www.linkedin.com/in/moin-dalvi-277b0214a//"><img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/></a>\
</p>\
</td>\
</tr> \
  </table>', unsafe_allow_html=True)

if __name__ == '__main__':
     main()
