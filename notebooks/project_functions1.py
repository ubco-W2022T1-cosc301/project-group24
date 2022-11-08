import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
def load_and_process(url_or_path_to_csv_file):
    

    # Method Chain 1 (Load data and deal with missing data)

    df1 = (
          pd.read_csv(url_or_path_to_csv_file)
          .dropna(axis='columns')
          .isnull().any(axis=1)
      )

    # Method Chain 2 (Create new columns, drop others, and do processing)

    df2 = (
          df1
            .drop(columns=["EmployeeCount",'EmployeeNumber','StandardHours','Over18'])
            
      )

    # Make sure to return the latest dataframe
    return df2 