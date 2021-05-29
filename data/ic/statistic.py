import pandas as pd
from pprint import pprint

df_train = pd.read_csv('./train.csv', encoding='utf-8-sig')

df_train = df_train[["sentence", "intent"]]

print('\nDistributed Statistic Train Dataset:\n')
pprint(df_train.groupby('intent').count())