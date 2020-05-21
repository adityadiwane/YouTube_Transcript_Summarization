import os
import glob
import re
import pdb
import csv
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

repl_mapping = {'\'ve': ' have','\'s':' is','\'re':' are','\'d':' would','n\'t' : ' not','\'ll':' will','\'m':' am','y\'':'you ','let\'s':'let us'}

def expand_word(word):
    for part,repl in repl_mapping.items():
        word = word.replace(part,repl)
    return word


def load_transform_data(file_path,**kwargs):
    fx_run_x = kwargs.get('fx_run_x')
    fx_run_y = kwargs.get('fx_run_y')
    fp = open(file_path + '\\ted_metadata_youtube.csv',encoding='utf-8')
    data_df = {'x' : [] , 'y' : []}
    reader = csv.DictReader(fp,delimiter=',',quotechar ='"')
    for i in reader:
        try:
            #pdb.set_trace()
            desc = i['description']
            desc = re.sub(r'[^\x00-\x7F]+','', desc)
            #desc = re.sub('')
            
            #pdb.set_trace()
            video_id =i['id']
            #flist = glob.glob('subtitles/'+id+'/'+id+'.en.vtt',recursive=True)
            
            #for f in flist:
            fname = file_path + r'\\tedDirector\\subtitles\\'+video_id+'\\'+video_id+'.en.vtt'
            if os.path.exists(fname):
                with open(fname,encoding='utf-8') as fp:
                    try:
                        #pdb.set_trace()
                        lines = fp.read()
                        y = fx_run_y(desc)
                        x = fx_run_x(lines)
                        if ( x or y ) != False:
                            data_df['x'].append(x)
                            data_df['y'].append(y)
                    except Exception as e:
                        print("Exception in subtitle:" , e)
        except Exception as e:
            print("Exception in csv:" , e)
    return data_df
    



