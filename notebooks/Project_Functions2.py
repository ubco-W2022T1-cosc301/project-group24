import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

def load_and_process(url_or_path_to_csv_file):
    df = pd.read_csv(url_or_path_to_csv_file)
    cleaned_df = df.drop(columns=["EmployeeCount","EmployeeNumber","StandardHours","Over18"])
    return cleaned_df

def EDA(df):
    FirstEda(df)
    SecondEda(df)
    
def FirstEda(df):
    display(df.info())
    display(df.head)
    display(df.shape)
    display(list(df.columns))
    display(df.describe())
    display(df.nunique())
    
def Info(df)
    print("info")
    display(df.info())
    
def SecondEda(df):
    corr = df.corr()
    sns.heatmap(corr, xticklabels = corr.columns, yticklabels=corr.columns, annot = False, cmap = sns.diverging_palette(220,20,as_cmap=True))
    hist = df.hist(bins = 10,figsize = (20,10))
