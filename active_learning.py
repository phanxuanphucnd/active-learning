import random
import numpy as np
import math
import datetime
import csv
import re
import os
import getopt, sys
import glob
import pandas as pd

from random import shuffle
from collections import defaultdict

from uncertainty_sampling import UncertaintySampling
from models.model import evaluate_model, train_model

from pprint import pprint

# annotate this many randomly sampled items first for evaluation data before creating training data
min_eval_items = 1000

# minimum number of training items before we first train a model
min_train_items = 1000


data = []
test_data = []

# directories with data
unlabeled_data_folder = "data/ic/unlabel_data/"

evaluation_data = "data/ic/test.csv"

training_data = "data/ic/train.csv"


# default number to sample for each method

number_random = 0

number_least_conf = 0
number_margin_conf = 0
number_ratio_conf = 0
number_entropy_based = 0

verbose = False

cli_args = sys.argv
arg_list = cli_args[1:]

# default option, random:
gnu_options = ["random_remaining="]

# uncertainty sampling
gnu_options += ["least_conf=", "margin_conf=", "ratio_conf=", "entropy_based=", "mode=", 
                "file=", "pretrained_model="]

# options
gnu_options += ["help", "verbose"]

try:
    arguments, values = getopt.getopt(arg_list, "", gnu_options)
except getopt.error as err:
    print(str(err))
    sys.exit(2)

file_unlabel = None
pretrained_model = None
mode = 0

for arg, value in arguments:
    if arg == "--random_remaining":
        number_random = int(value)
    if arg == "--least_conf":
        number_least_conf = int(value)
    if arg == "--margin_conf":
        number_margin_conf = int(value)
    if arg == "--ratio_conf":
        number_ratio_conf = int(value)
    if arg == "--entropy_based":
        number_entropy_based = int(value)
    if arg == "--mode":
        mode = int(value)
    if arg == "--pretrained_model":
        pretrained_model = value
    if arg == "--file":
        file_unlabel = value
    if arg == "--type":
        type_al = value
    if arg == "--verbose":
        verbose = True
    if arg == "--help":
        print("\nValid options for Active Learning sampling: ")
        for option in gnu_options:
            print("\t" + option)
        print("\n")
        exit()

already_labeled = {}

def load_data(filepath, skip_already_labeled=False):
    # csv format: [SENID, SENTENCE, INTENT, SAMPLING_STRATEGY, CONFIDENCE]

    with open(filepath, 'r') as csvfile:
        data = []
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i != 0:
                if skip_already_labeled and row[0] in already_labeled:
                    continue

                if len(row) < 3:
                    row.append("") # add empty col for INTENT to add later
                if len(row) < 4:
                    row.append("") # add empty col for SAMPLING_STRATEGY to add later
                if len(row) < 5:
                    row.append("") # add empty col for CONFIDENCE to add later
                data.append(row)

                label = str(row[2])
                if row[2] != "":
                    textid = row[0]
                    already_labeled[textid] = label

    csvfile.close()

    return data

def append_data(filepath, data):
    with open(filepath, 'a', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    csvfile.close()

def append_dataframe(df1, df2):
    '''
    Append two dataframes and drop duplications
    '''

    df = df1.append(df2)
    df = df.drop_duplicates(subset='senid', keep='first')

    df.to_csv('./data/ic/train.csv', encoding='utf-8-sig', index=False)

def writer_data(filepath, data):
    with open(filepath, 'w', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close()


# LOAD ALL UNLABELED, TRAINING, and EVAL DATA

train_data = load_data(training_data)
train_count = len(train_data)

print(f"\nThe number of samples Training data: {train_count}; Shape:  {np.shape(train_data)}\n")

eval_data = load_data(evaluation_data)
eval_count = len(eval_data)

print(f"The number of samples Evaluate data: {eval_count}; Shape: {np.shape(eval_data)}\n")

data = []

if file_unlabel is None:
    for file in glob.glob(unlabeled_data_folder + "*.csv"):
        data += load_data(file)
else:
    data = load_data(unlabeled_data_folder + file_unlabel)

## SELECT THE NUMBERS SAMPLES UNLABEL DATA
# data = data[:500]

classes = ['agree', 'ask_is_bot', 'connect_employee', 'disagree', \
    'greet', 'handover_to_inbox', 'query_knowledge_base', 'thank']

# Annotation instructions
annotation_instructions = "\nInstructions labeling for Intent Classification (IC) \n\n"
annotation_instructions += f"\tLabel options:\n"
annotation_instructions += f"\t         0           : if this sentence is in class `{classes[0]}`\n"
annotation_instructions += f"\t         1           : if this sentence is in class `{classes[1]}`\n"
annotation_instructions += f"\t         3           : if this sentence is in class `{classes[3]}`\n"
annotation_instructions += f"\t         2           : if this sentence is in class `{classes[2]}`\n"
annotation_instructions += f"\t         5           : if this sentence is in class `{classes[5]}`\n"
annotation_instructions += f"\t         4           : if this sentence is in class `{classes[4]}`\n"
annotation_instructions += f"\t         6           : if this sentence is in class `{classes[6]}`\n"
annotation_instructions += f"\t         7           : if this sentence is in class `{classes[7]}`\n"
annotation_instructions += f"\tOther options:\n"
# annotation_instructions += f"\t        -1           : to ignore samples,\n"
annotation_instructions += f"\t     back  or  b     : to go back to the last sentence,\n"
annotation_instructions += f"\t     detail  or  d   : to see detailed definitions,\n"
annotation_instructions += f"\t     save  or  s     : to save your annotations.\n"

last_instructions = f"\nALL DONE !\n\n"
last_instructions += f"Type:\n"
last_instructions += f"\t   back  or  b    : to go back to change any labels,\n"
last_instructions += f"\t   save  or  s     : to save your annotations.\n"

df = pd.read_csv("data/define_ic.csv", encoding="utf-8-sig")
detailed_instructions = "\nDescribe the definition of intent: \n\n"
for i in range(len(df)):
    intent = df["intent"][i]
    description = df["description"][i]
    space = " ".join(["" for s in range(25-len(intent))])
    detailed_instructions += f"> {intent}: {space} {description} \n\n"

def get_annotations(data, default_sampling_strategy="random"):
    """
    Prompts annotator for label from command line and adds annotations to data

    Keyword arguments:
        data: an list of unlabeled items where each item is
            [SENID, SENTENCE, INTENT, SAMPLING_STRATEGY, CONFIDENCE]
        default_sampling_strategy: strategy to use for each item if not already specified
    """

    ind = 0
    while ind <= len(data):
        if ind < 0:
            ind = 0 # in case didn't go back before the first
        if ind < len(data):
            textid = data[ind][0]
            text = data[ind][1]
            label = data[ind][2]
            strategy = data[ind][3]
            score = data[ind][4]

            if strategy == "":
                strategy = random

            if textid in already_labeled:
                if verbose:
                    print(f"Skipping seen `{str(textid)}` with label `{label}`")
                    print(data[ind])
                ind += 1
            else:
                print(annotation_instructions)
                if verbose:
                    print(f"Sampled with strategy `{str(strategy)}` and score `{str(round(score, 3))}`")

                label = str(input("TEXT:\t" + text + " | score: " + str(score) + "\n> Enter LABEL:\t"))

                if label == "back" or label == "b":
                    ind -= 1  # go back the last sentence
                elif label == "detail" or label == "d":
                    print(detailed_instructions)  # print detailed instructions
                elif label == "-1":
                    ## ignore sample
                    break
                elif label == "save" or label == "s":
                    break
                else:
                    label = classes[int(label)]

                    data[ind][2] = label  # add label to our data
                    if data[ind][3] is None or data[ind][3] == "":
                        data[ind][3] = default_sampling_strategy  # add default if none given
                    ind += 1
        else:
            # last one - give annotator a change to go back
            print(last_instructions)
            label = str(input("> Enter:\t"))
            if label == "back" or label == "b":
                ind -= 1
            else:
                ind += 1

    return data

def get_random_items(unlabeled_data, number=10):
    shuffle(unlabeled_data)

    random_items = []
    for item in unlabeled_data:
        text_id = item[0]
        if text_id in already_labeled:
            continue
        item[3] = "random_remaining"
        random_items.append(item)
        if len(random_items) >= number:
            break

    return random_items

#================================ ACTIVE LEARNING ========================================#


print("\nStarting...\n\n")

if eval_count <  min_eval_items:
    #Keep adding to evaluation data first
    print("Creating evaluation data:\n")

    shuffle(data)
    needed = min_eval_items - eval_count
    data = data[:needed]
    print(f"Evaluate data: {str(needed)} more annotations needed")

    data = get_annotations(data) 

    # append evaluation data
    append_data(evaluation_data, data)

if train_count < min_train_items:
    # lets create our first training data! 
    print("Creating initial training data:\n")

    shuffle(data)
    needed = min_train_items - train_count
    data = data[:needed]
    print(f"Training data: {str(needed)} more annotations needed")

    data = get_annotations(data)
    
    # append training data
    append_data(training_data, data)
else:

    if mode == 0:
        # Lets start ACTIVE LEARNING
        sampled_data = []
        # Get random samples
        if number_random > 0:
            print(f"Sampling {str(number_random)} Random Remaining Items\n")
            sampled_data += get_random_items(data, number=number_random)

        model_path = ""
        # RETRAIN WHOLE MODEL IF WE NEED IT FOR ANY METHOD:
        if (number_least_conf + number_margin_conf + number_ratio_conf + number_entropy_based > 0):            

            if pretrained_model is None:
                print(f"> Retraining model for Uncertainty Sampling\n")
                model_path, mean, std = train_model(train_data, step=0)
                accuracies = evaluate_model(model_path, eval_data, mode='retrain')
                print(f"\nEVALUATE MODEL\nStep 1:\n")
                print(f"\n\tAccuracy={accuracies[0]} | Precision={accuracies[1]} | Recall={accuracies[2]} | F1-score={accuracies[3]}\n")
                print(f"\t> Model saved to: {model_path}\n")
            else:
                print(f"> Loading model for Uncertainty Sampling\n")
                model_path = pretrained_model
        
        uncert_sampling = UncertaintySampling(verbose)

        if number_least_conf + number_margin_conf + number_ratio_conf + number_entropy_based > 0:
            # Get least confidence samples
            if number_least_conf > 0:
                print(f" > Sampling {str(number_least_conf)} via Least Confidence Sampling\n")

                sampled_data += uncert_sampling.get_samples(model_path, data, uncert_sampling.least_confidence, number=number_least_conf)
            
            # Get margin of confidence samples
            if number_margin_conf > 0:
                print(f" > Sampling {str(number_margin_conf)} via Margin of Confidence Sampling\n")

                sampled_data += uncert_sampling.get_samples(model_path, data, uncert_sampling.margin_confidence, number=number_margin_conf)
            
            # Get ratio of confidence samples
            if number_ratio_conf > 0:
                print(f" > Sampling {str(number_ratio_conf)} via Ratio of Confidence Sampling\n")

                sampled_data += uncert_sampling.get_samples(model_path, data, uncert_sampling.ratio_confidence, number=number_ratio_conf)
                

            # Get entropy-based samples
            if number_entropy_based > 0:
                print(f" > Sampling {str(number_entropy_based)} via Entropy-based Sampling\n")

                sampled_data += uncert_sampling.get_samples(model_path, data, uncert_sampling.entropy_based, number=number_entropy_based)

        df_sampled = pd.DataFrame(sampled_data, columns=['senid', 'sentence', 'intent', 'sampling_strategy', 'confidence'])
        
        if not os.path.exists('./data/sampled_data/'):
            os.makedirs('./data/sampled_data/')
        
        df_sampled = df_sampled[df_sampled['confidence'] >= 0.4]

        # df_sampled = df_sampled[~df_sampled['senid'].isin(already_labeled) & df_sampled['confidence'] >= 0.4]
        # print(f"\n\n {df_sampled}")
        df_sampled.to_csv('./data/sampled_data/sampled_data.csv', index=False, encoding='utf-8-sig')
    
    
    # GET ANNOTATIONS FROM OUR SAMPLES
    # shuffle(sampled_data)

    # sampled_data = get_annotations(sampled_data)

    # append training data files
    # append_data(training_data, sampled_data)

if mode == 1:

    print(f"\n> Append Sampled-data into Training data...")

    sampled_data = load_data('./data/sampled_data/sampled_data.csv')
    append_data(training_data, sampled_data)
    print("\n---> Append data done!~\n")

    if train_count > min_train_items:
        print(f"\n> Retraining model with new data \n")

        # UPDATE OUR DATA AND (RE)TRAIN MODEL WITH NEWLY ANNOTATED DATA
        train_data = load_data(training_data)
        train_count = len(train_data)

        eval_data = load_data(evaluation_data)
        eval_count = len(eval_data)

        model_path, mean, std = train_model(train_data, step=1)

        accuracies = evaluate_model(model_path, eval_data, mode='new')
        print(f"\nEVALUATE MODEL")
        print(f"\n\tAccuracy={accuracies[0]} | Precision={accuracies[1]} | Recall={accuracies[2]} | F1-score={accuracies[3]}\n")
        #print(f"> Model saved to: {model_path}")

print("\nFinished...\n\n")
