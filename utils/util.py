import os
import re
import ast
import configparser
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from pandas.plotting import table
from unicodedata import normalize as nl

# Function to plot confusion matrics
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

path = os.path.abspath(os.getcwd())
print(path)

def load_data(file_name):
    
    """
    This function loads csv file as pandas.dataframe in train.py and eval.py.

    :param str file_name: File name to load
    :return dataframe df: Whole dataset as dataframe type

    """
    
    file_path = os.path.abspath(file_name)
    if not file_name.endswith(".csv"):
        raise RuntimeError("file %s is of invalid file format" % file_path)

    df = pd.read_csv(file_path, encoding='utf-8', error_bad_lines=False)
    return df

def normalize_df(df):
    
    # tokenizing
    tokenized_doc = df['text'].apply(lambda x: x.split())

    # remove stop-words
    # tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

    # de-tokenizing
    de_tokenized_doc = []
    
    for i in range(len(df)):
        t = ''
        try:
            t = ' '.join(tokenized_doc[i])
        except:
            print("Error")
            print(i)
        de_tokenized_doc.append(t)
    df['text'] = de_tokenized_doc
    return df


def prepare_df(path):
    """

    :param path:
    :return:
    """

    data_row = load_data(path)
    return normalize_df(data_row)

def tokenize(text):
    # return [x.lower() for x in nltk.word_tokenize(text)]
    return [ x.lower() for x in text.split() ]

def normalize(txt):
    
    '''
    
    Function to normalize the user's input
    :params:
        txt : str: The text to normalize
    :return:
        txt : str: The text after normalize
    
    '''

    txt = nl('NFKC', txt)

    # Remove emoji
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    txt = emoji_pattern.sub(r" ", txt)

    # Remove url, link
    url_regex = re.compile(r'\bhttps?://\S+\b')
    txt = url_regex.sub(r" ", txt)

    # Remove \n
    txt = re.sub(r"\n", ". ", txt)

    # Remove special token and duplicate
    txt = re.sub(r"[^a-z0-9A-Z*\.\,\?\!\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]", " ", txt)
    txt = re.sub(r"\?{2,}", "? ", txt)
    txt = re.sub(r"\!{2,}", "! ", txt)
    txt = re.sub(r"\.{2,}", ". ", txt)
    txt = re.sub(r"\,{2,}", ", ", txt)
    txt = re.sub(r"\s{2,}", " ", txt)

    return txt


def check_url_exists(txt):
    
    '''
    
    Function to check url exists. Return True if exists, otherwise
    :param:
        txt : str: The user's input
    :return:
        True if exists url, otherwise
    
    '''

    url_regex = re.compile(r'\bhttps?://\S+\b')

    return re.match(url_regex, txt) is not None


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, 
                          save_dir=None):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90 )
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(
        accuracy, misclass))

    # plt.savefig((save_dir + './results/evaluate/{}.png'.format(title)))
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
        
    save_path = './results/{}.png'.format(title)
    plt.savefig(save_path)

    return save_path
