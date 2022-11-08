import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def load_and_process(dataset):
    df = pd.read_csv(dataset)
    cleaned_df = df.drop(columns=["EmployeeCount","EmployeeNumber","StandardHours","Over18"])
    categoricalAttributes = ["BusinessTravel","Department","Education","EducationField","EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","OverTime","PerformanceRating","RelationshipSatisfaction","StockOptionLevel","WorkLifeBalance"]
    for attribute in categoricalAttributes:
        setAsCategory(cleaned_df, attribute)
    return cleaned_df

def setAsCategory(df, attribute):
  df[attribute] = df[attribute].astype('category')

def EDA(df):
    FirstEda(df)
    SecondEda(df)
    
def FirstEda(df):
    Info(df)
    display(df.head)
    display(df.shape)
    display(list(df.columns))
    display(df.describe())
    display(df.nunique())
    
def Info(df):
    print("info")
    display(df.info())
    
def SecondEda(df):
    corr = df.corr()
    sns.heatmap(corr, xticklabels = corr.columns, yticklabels=corr.columns, annot = False, cmap = sns.diverging_palette(220,20,as_cmap=True))
    hist = df.hist(bins = 10,figsize = (20,10))