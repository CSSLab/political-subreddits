from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from datetime import datetime
from numpy.linalg import norm
from tqdm.auto import tqdm
from glob import glob
import pandas as pd
import numpy as np
import subprocess
import sys
import os
import nltk
import re


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

lemma = WordNetLemmatizer()
stopwords = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    # Remove urls
    text = re.sub('http[s]?://\S+', '', text)
    # remove emojis
    text = emoji_pattern.sub(r'', text)
    token_words = word_tokenize(text)
    pos_tags = nltk.pos_tag(token_words)
    stem_text=[]
    for token,pos in pos_tags:
        token = re.sub("[>@,\.!?']", '', token)
        if token not in stopwords and len(token) > 3:
            pos_tag = get_wordnet_pos(pos)
            token = lemma.lemmatize(token,pos=pos_tag) if pos_tag else token 
            stem_text.append(token)
    return stem_text

       
if __name__ == "__main__":
    print(clean_text("He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."))
    

cos_sim = lambda a,b : np.dot(a, b)/(norm(a)*norm(b))
cos_dist = lambda a,b : 1 - cos_sim(a,b)

CANDIDATE_SUBS = ["JoeBiden","SandersForPresident","BaemyKlobaechar","ElizabethWarren","Pete_Buttigieg","YangForPresidentHQ"]

def generate_embedding(time_frame=None,**arg_dict):
    output = "./trained_embeddings/vecs_{p1}_{p2}.txt".format(**arg_dict)
    if time_frame:
        subprocess.run("mkdir -p trained_embeddings/temporal/{}".format(time_frame), shell=True)
        output = "./trained_embeddings/temporal/{}/{}_vecs_{p1}_{p2}.txt".format(time_frame,time_frame,**arg_dict)
    command = "./word2vecf/word2vecfadapted -output {} -train {file_data} -wvocab {file_wv} -cvocab {file_cv} -threads 150 -alpha {alpha} -size {size} -{param1} {p1} -{param2} {p2}".format(output,**arg_dict)
    if not os.path.exists(output):
        print("\t * Starting {}".format(output))
        subprocess.run(command, shell=True)
    return output

def load_embedding(filepath,split=True, **kwargs):
    embedding = pd.read_csv(filepath, sep=' ', header=None, skiprows=1, **kwargs)
    embedding.set_index(0)
    embedding = embedding.rename(columns={0: 'subreddit'})
    subreddits, vectors = embedding.iloc[:, 0], embedding.iloc[:, 1:151]
    vectors = vectors.divide(np.linalg.norm(vectors, axis=1), axis=0)
    if split:
        return subreddits, vectors
    embedding = pd.concat([subreddits, vectors], axis=1).set_index("subreddit")
    return embedding

def parse_tup(tup,date_str="%Y-%m-%d"):
    to_tup = tup.strip('()').split(',')
    to_tup[1] = datetime.strptime(to_tup[1],date_str)
    return to_tup

def coalese_csvs(dir_fp,output_fp,chunksize=50000):
    csv_file_list = glob("{}/*.csv".format(dir_fp)) 
    if not os.path.exists(output_fp):
        for csv_file_name in tqdm(csv_file_list):
            chunk_container = pd.read_csv(csv_file_name, chunksize=chunksize)
            for chunk in chunk_container:
                chunk.to_csv(output_fp, mode="a", index=False)
    return output_fp

