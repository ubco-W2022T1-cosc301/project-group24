import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from IPython.display import display

def load_and_process(url_or_path_to_csv_file):
  ### Loads the CSV file, cleans, and processes the dataframe ###
  df = pd.read_csv(url_or_path_to_csv_file)
  cleaned_df = df.copy().drop(columns=["EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"])
  categoricalAttributes = ["BusinessTravel", 
                           "Department", "Education", "EducationField", 
                           "EnvironmentSatisfaction", "Gender", 
                           "JobInvolvement", "JobLevel", "JobRole", 
                           "JobSatisfaction", "MaritalStatus", "OverTime",
                           "PerformanceRating", "RelationshipSatisfaction",
                           "StockOptionLevel", "WorkLifeBalance"]
  for attribute in categoricalAttributes:
    setAsCategory(cleaned_df, attribute)
  return cleaned_df

def setAsCategory(df, attribute):
  ### Sets an attribute to a category type for a dataframe ###
  df[attribute] = df[attribute].astype('category')

def eda(df):
  ### Calls all 3 EDA's (Basic, Categorical, and Numeric) ###
  basicEda(df)
  categoricalEda(df)
  numericalEda(df)
  
def basicEda(df):
  ### Produces the basic information about the dataframe ###
  printDfInfo(df)
  printDfHead(df)
  printDfShape(df)
  printDfColumns(df)
  printDfDescription(df)
  printDfUniques(df)
  
def categoricalEda(df):
  ### Runs an EDA for the categorical attributes in the dataframe ###
  categoricalAttributes = list(df.select_dtypes(include='category').columns)
  distributionByCategories(df, categoricalAttributes)
  categoriesSplitByHue(df, categoricalAttributes, "Attrition")
  printAttritionPercentages(df)
  
def numericalEda(df):
  ### Runs an EDA for the numerical attributes in the dataframe ###
  corrDf(df)
  attritionCorrDf(df)
  
def printDfInfo(df):
  ### Prints the dataframe information with title ###
  print("\n\nDataframe Basic Information: \n")
  display(df.info())
  
def printDfHead(df):
  ### Prints the dataframe head with title ###
  print("\n\nDataframe First 5 Rows: \n")
  display(df.head())
  
def printDfShape(df):
  ### Prints the dataframe shape with title ###
  print("\n\nDataframe Rows and Columns: ")
  display(df.shape)
  
def printDfColumns(df):
  ### Prints the dataframe columns with title ###
  print("\n\nColumns: ")
  display(list(df.columns))
  
def printDfDescription(df):
  ### Prints the dataframe description with title ###
  print("\n\nDescription of the numerical columns: ")
  display(df.describe())
  
def printDfUniques(df):
  ### Prints the dataframe unique values with title ###
  print("\n\nNumber of unique values for each column: ")
  display(df.nunique())
  
def printAttritionPercentages(df):
  ### Prints the percentage of attrition per category dataframe with title ###
  print("\n\nPercentage of Attrition across categorical columns:")
  attritionPercentageDf(df)
  
def attritionPercentageDf(df):
  ### Produces a table of the percentages of attrition per value in each categorical attribute ###
  percentages = list()
  for column in list(df.select_dtypes(include='category').columns):
    for value in df[column].unique():
      data = df[df[column] == value]["Attrition"].value_counts()
      percentage = data["Yes"] / (data["Yes"] + data["No"])
      percentages.append([column, value, round(percentage, 2)])
  noIndex = [''] * len(percentages)
  percentageDf = pd.DataFrame(percentages, columns=["Category", "Value", "Percentage"], index=noIndex)
  percentageDf = percentageDf.sort_values(by=["Percentage"], ascending=False)
  display(percentageDf)
  
def corrDf(df):
  ### Produces a heatmap of the correlations between the numerical attributes ### 
  labels, categories = pd.factorize(df["Attrition"])
  df["Attrition"] = labels
  corr = df.corr()
  
  fig, axs = plt.subplots(1, 1, figsize=(8, 7))
  sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
  
def attritionCorrDf(df):
  ### Produces a table of the correlations between attrition and the numerical attributes ###
  labels, categories = pd.factorize(df["Attrition"])
  df["Attrition"] = labels
  corr = df.corr()
  
  print("\n\nCorrelation between Numerical Attributes and Attrition ordered by correlation magnitude:")
  topCorr = pd.DataFrame.from_dict(corr["Attrition"].abs())
  topCorr.rename(columns={"Attrition": "Correlation"}, inplace=True)
  topCorr = topCorr.sort_values(by=["Correlation"], ascending=False)
  display(topCorr)
  
def distributionByCategories(df, attributes):
  ### Produces multiple subplots for the distribution of each categorical attribute ###
  subplot_rows = int(math.ceil(len(attributes) / 3))
  subplot_cols = 3
  fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(5*subplot_cols,3*subplot_rows))
  fig.suptitle("Distribution by Categories", y=0.92, fontsize=24)
  plt.subplots_adjust(wspace=0.6, hspace=0.4)
  
  for row in range(subplot_rows):
    for col in range(subplot_cols):
      index = row*subplot_cols + col
      if index >= len(attributes):
        break
      distributionByCategory(df, attributes[index], axs[row, col])

def distributionByCategory(df, attribute, ax):
  ### Produces one plot for the distribution of a categorical attribute ###
  unique = list(df[attribute].unique())
  unique.sort(reverse=True)
  plot = sns.countplot(data=df, y=attribute, order=unique, ax=ax)
  plot.set_yticklabels(plot.get_yticklabels(), rotation=50)
  plot.set(title=f"Participant Distribution By {attribute}", xlabel="Count", ylabel=attribute)
  
  
def categoriesSplitByHue(df, attributes, hue):
  ### Produces multiple subplots for the each categorical attribute split by a hue ###
  subplot_rows = int(math.ceil(len(attributes) / 3))
  subplot_cols = 3
  fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(5*subplot_cols,3*subplot_rows))
  fig.suptitle(f"Categories split by {hue}", y=0.92, fontsize=24)
  plt.subplots_adjust(wspace=0.6, hspace=0.4)
  
  for row in range(subplot_rows):
    for col in range(subplot_cols):
      index = row*subplot_cols + col
      if index >= len(attributes):
        break
      categorySplitByHue(df, attributes[index], hue, axs[row, col])


def categorySplitByHue(df, attribute, hue, ax):
  ### Produces one plot for a categorical attribute split by a hue ###
  unique = list(df[attribute].unique())
  unique.sort(reverse=True)
  plot = sns.countplot(data=df, y=attribute, order=unique, hue=hue, ax=ax)
  plot.set_yticklabels(plot.get_yticklabels(), rotation=50)
  plot.set(title=f"{attribute} split by {hue}", xlabel="Count", ylabel=attribute)
  