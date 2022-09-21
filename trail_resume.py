import pandas as pd
import numpy as np
import streamlit as st
import docx2txt, pdfplumber, re, os, docx, PyPDF2, spacy, nltk
# load pre-trained model
nlp = spacy.load('en_core_web_sm')
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot  as plt
stop=set(stopwords.words('english'))
from pickle import load
import aspose.words as aw
from spacy.matcher import Matcher
# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

model=load(open('C:/Users/Moin Dalvi/Downloads/Randomforest.sav','rb'))
model1 = load(open('C:/Users/Moin Dalvi/Downloads/tfidf.pkl','rb'))

nltk.download('wordnet')
nltk.download('stopwords')

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
    data = pd.read_csv("C:/Users/Moin Dalvi/skills.csv") 
            
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
    if filename.endswith('.docx'):
        doc = docx.Document(filename)
        
        for para in doc.paragraphs:
            fullText = fullText + para.text
            
           
    elif filename.endswith('.pdf'):  
        with open(filename, "rb") as pdf_file:
            pdoc = PyPDF2.PdfFileReader(filename)
            number_of_pages = pdoc.getNumPages()
            page = pdoc.pages[0]
            page_content = page.extractText()
             
        for paragraph in page_content:
            fullText =  fullText + paragraph
            
    else:
        import aspose.words as aw
        output = aw.Document()
        # Remove all content from the destination document before appending.
        output.remove_all_children()
        input = aw.Document(filename)
        # Append the source document to the end of the destination document.
        output.append_document(input, aw.ImportFormatMode.KEEP_SOURCE_FORMATTING)
        output.save("Output.docx");
        doc = docx.Document('Output.docx')
        
        for para in doc.paragraphs:
            fullText = fullText + para.text
        fullText = fullText[79:]
         
    return (fullText)

def display(doc_file):

    
    resume = []
    if doc_file.endswith('.docx'):
        resume.append(docx2txt.process(doc_file))

    elif doc_file.endswith('.pdf'):
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())

    else:
        fullText = ''
        output = aw.Document()
        # Remove all content from the destination document before appending.
        output.remove_all_children()
        inputs = aw.Document(doc_file)
        # Append the source document to the end of the destination document.
        output.append_document(inputs, aw.ImportFormatMode.KEEP_SOURCE_FORMATTING)
        output.save("Output.docx");
        doc = docx.Document('Output.docx')
        
        for para in doc.paragraphs:
            fullText = fullText + para.text
        resume.append(fullText[79:])
            
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
    
    # following lines create boxes in which user can enter the absolute directory 

    with st.form(key="form1"):
        st.warning(body="Supported file Formats: '.docx' or '.doc' or '.pdf'")
        folder_pathe=st.text_input(label= "Enter the absolute folder path below in the box")
        st.markdown("Eg.  C:\\Users\\Admin\\Documents\\foldername consisting resumes")
        submit = st.form_submit_button(label="Click Me!")                           

        if submit:

            for files in os.listdir(folder_pathe):
                if files.endswith('.docx') or files.endswith('.doc') or files.endswith('.pdf'):
                    file_pathe  = os.path.join(folder_pathe, files)
                    filename.append(files)
                    cleaned=preprocess(display(file_pathe))
                    prediction = model.predict(model1.transform([cleaned]))[0]
                    predicted.append(target.get(prediction))
                    extText = getText(file_pathe)
                    exp = expDetails(extText)
                    experience.append(exp)
                    skills.append(extract_skills(extText))
    if len(predicted) > 0:
        file_type['Uploaded File'] = filename
        file_type['Experience'] = experience
        file_type['Skills'] = skills
        file_type['Predicted Profile'] = predicted
        file_type
        # Custom formatting
        # st.table(file_type.style.format({'Experience': '{:.1f}'}))

    st.subheader("About") 
    st.info("This project is a part of AiVariant Internship")

if __name__ == '__main__':
     main()