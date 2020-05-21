import nltk
import os
import re
import math
import operator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from clean_data import load_transform_data

import pdb

nltk.download('averaged_perceptron_tagger')
Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()
def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
       lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words
def stem_words(words):
    stemmed_words = []
    for word in words:
       stemmed_words.append(stemmer.stem(word))
    return stemmed_words
def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text
def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
       if word not in words_unique:
           words_unique.append(word)
    for word in words_unique:
       dict_freq[word] = words.count(word)
    return dict_freq
def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
             pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb
def tf_score(word,sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf
def idf_score(no_of_sentences,word,sentences):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if word.lower() not in Stopwords and len(word)>1]
        sentence = [word.lower() for word in sentence]
        sentence = [wordlemmatizer.lemmatize(word) for word in sentence]
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences/no_of_sentence_containing_word)
    return idf
def tf_idf_score(tf,idf):
    return tf*idf
def word_tfidf(dict_freq,word,sentences,sentence):
    word_tfidf = []
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf
def sentence_importance(sentence,dict_freq,sentences):
     sentence_score = 0
     sentence = remove_special_characters(str(sentence)) 
     sentence = re.sub(r'\d+', '', sentence)
     pos_tagged_sentence = [] 
     no_of_sentences = len(sentences)
     pos_tagged_sentence = pos_tagging(sentence)
     for word in pos_tagged_sentence:
          if word.lower() not in Stopwords and word not in Stopwords and len(word)>1: 
                word = word.lower()
                word = wordlemmatizer.lemmatize(word)
                sentence_score = sentence_score + word_tfidf(dict_freq,word,sentences,sentence)
     return sentence_score

def get_summary(text,threshold):
    tokenized_sentence = sent_tokenize(text)
    text = remove_special_characters(str(text))
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text)
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words)
    word_freq = freq(tokenized_words)
    no_of_sentences = int(threshold * len(tokenized_sentence))
    print(no_of_sentences)
    c = 1
    sentence_with_importance = {}
    for sent in tokenized_sentence:
        sentenceimp = sentence_importance(sent,word_freq,tokenized_sentence)
        sentence_with_importance[c] = sentenceimp
        c = c+1
    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)
    cnt = 0
    summary = []
    sentence_no = []
    for word_prob in sentence_with_importance:
        if cnt < no_of_sentences:
            sentence_no.append(word_prob[0])
            cnt = cnt+1
        else:
            break
    sentence_no.sort()
    cnt = 1
    #pdb.set_trace()
    summary = ". ".join([tokenized_sentence[i - 1] for i in sentence_no])
    return summary , no_of_sentences
    #print({tokenized_sentence[i-1 ]:j for i,j in sentence_with_importance})
    #print(len(tokenized_sentence))

def clean_x(string):
    string = preprocess(''.join(list(filter(lambda x: '-->' not in x and x != '',string.split('\n')))[3:]))
    #delimiter = '.'
    #pdb.set_trace()
    #update_ngrams(string,'x',delimiter)

    #pdb.set_trace()
    return string
    
def clean_y(string):
    string = preprocess(string)
    #delimiter = '.'
    #update_ngrams(string,'y',delimiter)
    
    #pdb.set_trace()
    return string
      
def preprocess(text):
    newString = text.lower()
    newString = re.sub('\b',' ', newString)
    newString = re.sub(r'\([^)]*\)', ' ', newString)
    newString = re.sub('"','', newString)
    newString = re.sub('http://[\w\/\.]+',' ', newString)
    newString = re.sub('[^\w.]',' ', newString)
    newString = re.sub('[.]','. ', newString)
    newString = ' '.join([expand_word(t) for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    #newString = re.sub("[^a-zA-Z]", " ", newString) 
    return newString

def text_clean(string):
    tokens = [w for w in string.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    string = (" ".join(long_words)).strip()
    return string

repl_mapping = {'\'ve': ' have','\'s':' is','\'re':' are','\'d':' would','n\'t' : ' not','\'ll':' will','\'m':' am','y\'':'you ','let\'s':'let us'}

def expand_word(word):
    for part,repl in repl_mapping.items():
        word = word.replace(part,repl)
    return word
    
    
def main():
    file_path = r'L:\Lectures\Semester_2\ML_Statistics\Project'
    data_df = load_transform_data( file_path , fx_run_x = clean_x, fx_run_y = clean_y )
    data_df['x'] = data_df['x']
    data_df['y'] = data_df['y']    
    #data_df['x'] = [ text_clean(i) for i in data_df['x']]
   
    #pdb.set_trace()
    fp = open('output.txt','w')
    for i in data_df['x']:
        print('-'*100)
        print('TEXT:'+i)
        print('-'*100)
        #pdb.set_trace()
        summary,no_of_sentences = get_summary(i,0.25)
        print('SUMMARY:' + summary)
        print('-'*100)
        fp.write('-'*100 + '\n')
        fp.write('Original Text: '+i)
        fp.write('-'*100 + '\n')
        fp.write('no_of_sentences '+ str(no_of_sentences) +' \n')
        fp.write(summary+'\n')
        fp.write('-'*100+ '\n')
    
    fp.close()
    
    
 
if __name__ == "__main__":
    main()