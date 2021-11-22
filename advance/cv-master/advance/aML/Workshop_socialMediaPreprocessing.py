#Sample Code to Do Text Pre Processing
# Walk Through all required Libraries in advance -
"""
Presented By:
Mehdi Zadeh
Resume Pages: https://www.linkedin.com/in/mehdihabibzadeh/
ßEmail: Zadeh1980mehdi@gmail.com
"""

numberOfInput=100
fileName='cleansocialMediaTexts.csv'
socialMediaRawText="./SampleSocialMediaText.csv"

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import pandas as pd
import re
# import spacy
# from spacy.tokens import Doc, Span
# from spacy_langdetect import LanguageDetector
import os
import imp
import sys
import numpy as np
import csv
from langdetect import detect
from langdetect import DetectorFactory

DetectorFactory.seed = 0

sys.modules["sqlite"] = imp.new_module("sqlite")
sys.modules["sqlite3.dbapi2"] = imp.new_module("sqlite.dbapi2")

################
sys.path.append('..../.ekphrasis')

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag, WordNetLemmatizer


from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk import wordpunct_tokenize

import nltk

import io
nltk.data.path.append(f"{os.getcwd()}/nltk_data")




df =  pd.read_csv(socialMediaRawText)
dfsocialMediaText=df[["socialMediaText"]][:numberOfInput]


# Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)


def languageCheck(text):

    nlp.add_pipe(LanguageDetector(language_detection_function=custom_detection_function), name="language_detector", last=True)

    doc = nlp(text)

    print("Likely average language of document : \n  ",doc._.language, "\n")

def leftoversCleanText(text): # in process!!

    #Check if it received anything
    if not text:
        return ''

    #get rid of non ascii characters
    #normalize#    text = normalize("NFKD", str(text)).strip() #TODO: check NFKD and see the parameters that work

    text = re.sub(r"([a-zA-Z]+)\?s ", r"\1 's ", text)             # special case toto?s => toto 's
    text = re.sub(r"('s|'d)", r' \1', text)                        # this one makes the other redundant
    text = re.sub(r'([$><+@{}!?()/;,%\^\[\]-])', r' \1 ', text)    # Separete this characters from text? -> text ?
    text = re.sub(r"([\-:$><+@{}!?()/;,%\^])",r" \1 ", text)       # these two do basically the same
    text = re.sub(r'([^&])([&])([^&])', r'\1 \2 \3', text)         #P&G -> P & G
    text = re.sub(r'&quot;|&lt;|&gt;|&lsquo;|&rsquo;|&ldquo;|&rdquo;|&nbsp;|&amp;|&apos;|&cent;|&pound;|&yen;|&euro;|&copy;|&reg;', ' ', text)
    text = re.sub(r"[\(|\{|\[]\s*?[\)|\}|\]]", ' ', text)          # (   ) -> '' removes empty brackets
    text = re.sub(r"'(?!(s|d))|^'", ' ', text)                     #'f -> ' ' removes apostrophe+letter that are not 's or 'd
    text = re.sub(r"\s+", " ", text)                               # change multiple blank spaces by just one
    #after tweepy preprocessing the colon symbol left remain after #removing mentions
    text = re.sub(r":", " ", text)
    text = re.sub(r"‚Ä¶", " ", text)

    replace consecutive non-ASCII characters with a space

    ###### Depends on case ########
    text = re.sub(r'[^\x00-\x017F]+',' ', text)
    #text = re.sub(r'[^\x00-\x7F]+',' ', text)

    #remove emojis from socialMediaText
    text = emoji_pattern.sub(r'', text)


    #remove symbols from text
    patternCommonRegularExpression = re.compile(r"\{|\}|\:|\\|/|\[|\]|\+|\<|\>|\_\•|\®|\*|\"|\“|\”|\!|\^|\↑|\❏|\$|\--|\|")
        #r"\:|\\|/|\[|\]|\+|\<|\>|\_\•|\®|\*|\"|\“|\”|\!|\?|\^|\↑|\❏|\$|\--|\&|\|\#|")



    text = patternCommonRegularExpression.sub("", text)  #


    return text


def replaceIsupperCapitalize(text):  # do this at very end!!!
    tokens = tokenize_text(text)
    filtered_text=[]

    for token in tokens:

        if not token.isupper() or len(token)<4:
            filtered_text.append(token)
        else:
            filtered_text.append(token.capitalize())


    filtered_text = ' '.join(filtered_text)
    return filtered_text


def leftoversTwitter(text):

    patternHappyCommonRegularExpression = re.compile(r" \＼\(\^o\^\)\／|\:\-\)|\:\)|\;\)|\:o\)| \:\]| \:3| \:c\)| \:\>|\=\]|8\)| \=\)| \:}|\:\^\)| \:\-D| \:D|8\-D|8D|x\-D|xD|X\-D|XD| \=\-D| \=D|\=\-3| \=3| \:\-\)\)| \:\'\-\)| \:\'\)| \:\*| \:\^\*| \>\:P| \:\-P| \:P|X\-P|x\-p| xp| XP|\:\-p|\:p|\=p|\:\-b|\:b| \>\:\)| \>\;\)| \>\:\-\)|\<3 ")
    text = patternHappyCommonRegularExpression.sub("", text)  #
    patternSadCommonRegularExpression = re.compile(r"\=/\/\|;\(|>\:?\\*|\:\{|\:c|\:\-c|\:'\-\(|>.*?<|:\(|>\:\(|=\/|\:L|\:-/|\>:/|\:S|\:\[|\:\-\|\|\:\-\)|\:\-\|\||\=L|\:<|\:\-\[|\:\-<|=/\/\|=\/|>\:\(|\:\(|\:'\-\(|\:'\(|\:?\\*|\=?\\?")
    text = patternSadCommonRegularExpression.sub("", text)  #

    return text
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

cleanTextList=[]

def preprocessingsocialMediaTexter(text):
    try:
        if detect(text) =='en':
            cleanText=text_processor.pre_process_doc(text)
            #print(type(cleanText))
            cleanText=' '.join(cleanText)
            #print(cleanText)

            cleanText=re.sub('<[^>]+>', '', cleanText)
            cleanText=re.sub(' +', ' ', cleanText).strip()
            cleanText=leftoversTwitter(cleanText)
            cleanText=leftoversCleanText(cleanText)
            cleanText = re.sub("\…", "", cleanText)

        #cleanText=replaceIsupperCapitalize(cleanText)
        else:
            cleanText="Text is NOT in English "
    except:
        cleanText="We had Empty input"

    return cleanText
    #Ref:https://stackoverflow.com/questions/26886653/pandas-create-new-column-based-on-values-from-other-columns-apply-a-function-o



###########################


dfsocialMediaText['CleanedsocialMediaTexts'] = dfsocialMediaText.apply(lambda row: preprocessingsocialMediaTexter(row['socialMediaText']), axis=1)

listcleansocialMediaTexts=dfsocialMediaText['CleanedsocialMediaTexts'].to_list()
print(dfsocialMediaText['CleanedsocialMediaTexts'][:5])
print(listcleansocialMediaTexts)
with open(fileName, 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(listcleansocialMediaTexts)

##########
dfsocialMediaText['CleanedsocialMediaTexts'].to_csv("./cleanedsocialMediaText2.csv", header=None, index=False)
