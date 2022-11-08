import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def load_and_process(url_or_path_to_csv_file):
    df = pd.read_csv(url_or_path_to_csv_file)
    cleaned_df = df.copy().drop(columns=["EmployeeCount","EmployeeNumber","StandardHours","Over18"])
    return cleaned_df

def eda(df):
    FirstEda(df)
    SecondEda(df)

    
def FirstEda(df):
    display(df.info())
    display(df.head)
    display(df.shape)
    display(list(df.columns))
    display(df.describe.apply(lambda s: s.apply(lambda x: format (x, 'f'))))
    display(df.nunique())
    
def SecondEda(df):
    
