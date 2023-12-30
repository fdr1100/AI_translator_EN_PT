import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.text_splitter import CharacterTextSplitter
import datetime
import os
import sys

############# Displaying images on the front end #################
st.set_page_config(page_title="Your AI translation App",
                   page_icon='♾️',
                   layout="centered",  #or wide
                   initial_sidebar_state="expanded",
                   menu_items={
                        'Get Help': 'https://docs.streamlit.io/library/api-reference',
                        'Report a bug': "https://www.extremelycoolapp.com/bug",
                        'About': "# This is a header. This is an *extremely* cool app!"
                                },
                   )

# 🈚🆗✅💬🇮🇹🇺🇸
#LOCAL MODEL EN-PT
#---------------------------------
#  UNICAMP Model
Model_PT = './model_en_pt/'   
#---------------------------------

### HEADER section
st.title("Your AI powered Text Translator 💬 ")
st.header("Translate your English text to Portuguese")
#st.image('Headline.jpg', width=750)
English = st.text_area("Paste here the English text...", height=300, key="original")
col1, col2, col3 = st.columns([2,5,2])
btn_translate = col2.button("✅ Start Translation", use_container_width=True, type="primary", key='start')
if btn_translate:
    if English:
        Model_PT = './model_en_pt/'   #torch
        with st.spinner('Initializing pipelines...'):
            st.success(' AI Translation started', icon="🆗")
            from langchain.text_splitter import CharacterTextSplitter
            # TEXT SPLITTER FUNCTION FOR CHUNKING
            text_splitter = CharacterTextSplitter(        
                separator = "\n\n",
                chunk_size = 300,
                chunk_overlap  = 0,
                length_function = len,
            )
            # CHUNK THE DOCUMENT
            st.success(' Chunking text...', icon="🆗")
            texts = text_splitter.create_documents([English])
            #print('[bold red] Inizialize AI toknizer...')
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            # INITIALIZE TRANSLATION FROM ENGLISH TO PORTUGUESE       
            tokenizer_tt0pt = AutoTokenizer.from_pretrained(Model_PT)  #google/byt5-small   #facebook/m2m100_418M
            st.success(' Initializing AI Model & pipeline...', icon="🆗")
            model_tt0pt = AutoModelForSeq2SeqLM.from_pretrained(Model_PT)  #Helsinki-NLP/opus-mt-en-it  or #Helsinki-NLP/opus-mt-it-en
            #print("pipeline")
            TToPT = pipeline("translation", model=model_tt0pt, tokenizer=tokenizer_tt0pt)
            # ITERATE OVER CHUNKS AND JOIN THE TRANSLATIONS
            finaltext = ''
            start = datetime.datetime.now() #not used now but useful
            print('[bold yellow] Translation in progress...')
            for item in texts:
                line = TToPT(item.page_content)[0]['translation_text']
                finaltext = finaltext+line+'\n'
            stop = datetime.datetime.now() #not used now but useful
            elapsed = stop - start
            st.success(f'Translation completed in {elapsed}', icon="🆗")
            print(f'[bold underline green1] Translation generated in [reverse dodger_blue2]{elapsed}[/reverse dodger_blue2]...')
            st.text_area(label="Translated text in Portuguese:", value=finaltext, height=350)
            st.markdown(f'Translation completed in **{elapsed}**')
            st.markdown(f"Translated number **{len(English.split(' '))}** of words")

    else:
        st.warning("You need some text to be translated!", icon="⚠️")