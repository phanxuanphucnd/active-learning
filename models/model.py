# import libraries
import sys
print(sys.path)
import fastai
from fastai import *
from fastai.text import (
    TextList, language_model_learner, AWD_LSTM, text_classifier_learner, TextLMDataBunch, load_learner,
    accuracy, FBeta, Precision, Recall
)
import pandas as pd
import numpy as np
import os
import glob
import datetime
import re
from utils.util import normalize
from sklearn.model_selection import KFold, StratifiedKFold
from pprint import pprint
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
import os
import torch
from unicodedata import normalize as nl
path = os.path.abspath(os.getcwd())

from utils.util import *

version = 'ic'
today = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")

CUDA = torch.cuda.is_available()
torch.cuda.manual_seed(1)
torch.cuda.set_device(0)

# torch.backends.cudnn.deterministic = True #tested - needed for reproducibility
# torch.backends.cudnn.benchmark = False

def train_model(train_data, step):
    """
    Train model on the given data
    """
    
    train = pd.DataFrame(train_data, columns=['senid', 'sentence', 'intent', 'sampling_strategy', 'confidence'])
    train_df = pd.DataFrame({
        'label': train.intent,
        'text': train.sentence
    })

    train_df = normalize_df(train_df)
    train_df = train_df.drop_duplicates(keep='first')

    print('Shape of Data Train: {} \n'.format(train_df.shape))

    dest = 'data/babe_unlabel/'

    data = (TextList.from_folder(dest)
                    .split_by_rand_pct(0.1, seed=42)
                    .label_for_lm()
                    .databunch(bs=48, num_workers=1))


    lm_fns = [path + '/models/lm/vi_wt1',  path + '/models/lm/vi_wt_vocab1']
    # learn_lm = language_model_learner(data, AWD_LSTM, pretrained_fnames=lm_fns, drop_mult=0.3)

    # learn_lm.unfreeze()
    # learn_lm.fit_one_cycle(1, 5.75e-02, moms=(0.8,0.7))

    data_lm = TextLMDataBunch.from_df(train_df = train_df, valid_df = train_df, path = "")
    print(f"Length vocab: {len(data_lm.vocab.itos)}")

    learn_lm = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=lm_fns, drop_mult=0.5)

    bs=128
    lr = 1e-3
    lr *= bs/48

    learn_lm.fit_one_cycle(2, lr*10, moms=(0.8,0.7))

    learn_lm.unfreeze()
    learn_lm.fit_one_cycle(8, lr, moms=(0.8,0.7))

    learn_lm.save(path +'/models/lm/vifine_tuned')
    learn_lm.save_encoder(path + '/models/lm/vifine_tuned_enc')

    ### Classifier

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    acc_val, f1_val, precision_val, recall_val = [], [], [], []

    # data_fold = (TextList.from_df(df_train, path, vocab=data_lm.vocab, cols='text')
    #              .split_by_rand_pct(0.1, seed=42)
    #              .label_from_df(cols='label')
    #              .databunch(bs=128, num_workers=1))

    if os.path.exists('./models/ic/{}_latest.pkl'.format(version)):
        os.rename('./models/ic/{}_latest.pkl'.format(version), './models/ic/{}_{}.pkl'.format(version, str(today)))

    model_path = './models/ic/{}_latest.pkl'.format(version)

    for train_id, val_id in skf.split(train_df, train_df['label']):
        data_fold = (TextList.from_df(train_df, path, vocab=data_lm.vocab, cols='text')
                    .split_by_idxs(train_id, val_id)
                    .label_from_df(cols='label')
                    .databunch(bs=128, num_workers=1))
        learn_c = text_classifier_learner(data_fold, AWD_LSTM, drop_mult=0.5, 
                                        metrics=[accuracy, 
                                                FBeta(average='weighted', beta=1), 
                                                Precision(average='weighted'),
                                                Recall(average='weighted')
                                                ])
        learn_c.load_encoder(path + '/models/lm/vifine_tuned_enc')
        learn_c.freeze()

        lr = 2e-2
        lr *= bs / 48
        ## 1
        learn_c.fit_one_cycle(3, lr, moms=(0.8, 0.7))
        ## 2
        learn_c.fit_one_cycle(3, lr, moms=(0.8,0.7))
        ## 3
        learn_c.freeze_to(-2)
        learn_c.fit_one_cycle(1, slice(lr / (2.6 ** 4), lr), moms=(0.8, 0.7))
        ## 4
        learn_c.freeze_to(-3)
        learn_c.fit_one_cycle(1, slice(lr / 2 / (2.6 ** 4), lr / 2), moms=(0.8, 0.7))
        ## 5
        learn_c.unfreeze()
        learn_c.fit_one_cycle(6, slice(lr / 10 / (2.6 ** 4), lr / 10), moms=(0.8, 0.7))
        
        pprint(learn_c.validate())
        results = learn_c.validate()
        acc_val.append(results[1])
        f1_val.append(results[2])
        precision_val.append(results[3])
        recall_val.append(results[4])

        if not os.path.exists('./models/ic/'):
            os.makedirs('./models/ic/')

        model_path = './models/ic/{}_latest.pkl'.format(version)
        learn_c.export(model_path)

    mean_acc = np.mean(acc_val)
    mean_precision = np.mean(precision_val)
    mean_recall = np.mean(recall_val)
    mean_f1 = np.mean(f1_val)
    print("\nMean | Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(mean_acc, mean_precision, mean_recall, mean_f1))

    std_acc = np.std(acc_val)
    std_precision = np.std(precision_val)
    std_recall = np.std(recall_val)
    std_f1 = np.std(f1_val)
    print("Std | Accuracy: {}, Precision: {}, Recall: {}, F1: {}\n\n".format(std_acc, std_precision, std_recall, std_f1))


    return model_path, [mean_acc, mean_precision, mean_recall, mean_f1], [std_acc, std_precision, std_recall, std_f1]

def evaluate_model(model_path, eval_data, mode='new'):
    """
    Evaluate the model on the held-out evaluation data

    Return the F-Score, AUC and Confusion Matrix
    """

    eval_df = pd.DataFrame(eval_data, columns=['senid', 'sentence', 'intent', 'sampling_strategy', 'confidence'])

    labels = eval_df['intent']
    predicteds = []
    scores = []

    path = "./"
    learn_c = load_learner(path, model_path)
    classes = learn_c.data.classes

    for i in range(len(eval_df)):
        text = normalize(eval_df['sentence'][i])

        out = learn_c.predict(text, with_dropout=True)
        pred = out[1]
        predicteds.append(classes[pred])
        scores.append(out[2][out[1]])

    eval_df["pred"] = predicteds
    eval_df["score"] = scores

    # metrics
    acc = accuracy_score(eval_df["intent"], eval_df["pred"])
    recall = recall_score(eval_df["intent"], eval_df["pred"], average='weighted')
    precision = precision_score(eval_df["intent"], eval_df["pred"], average='weighted')
    f1 = f1_score(eval_df["intent"], eval_df["pred"], average='weighted')

    cm = confusion_matrix(labels, predicteds, classes)
    file_path = plot_confusion_matrix(cm, normalize=True, target_names=classes, title="cm {} 1 {} {}".format(version, today, mode))
    # file_path = plot_confusion_matrix(cm, normalize=False, target_names=classes, title="cm {} ic 2 {}".format(version, today))

    print(f"\nConfusion matrix saved to: {file_path}\n")

    return [acc, precision, recall, f1]
