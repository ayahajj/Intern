#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Reading Data

# In[2]:


# Read the Titanic Dataset
df = pd.read_csv('C:/Users/USER/Downloads/Titanic-Dataset.csv')


# In[3]:


#Display the first rows of the dataset
df.head()


# In[4]:


df.tail()


# EDA - Exploratory Data Analysis

# In[5]:


shape = df.shape
print(" - The dataset contains", shape[1] ,"columns and" , shape[0] , "rows.")


# In[6]:


column_names = df.columns
print(" - The column names are:")
for col in column_names:
    print("    -", col)


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


#check missing values
print(" - Total NAs: ", df.isna().sum().sum())
print()
print(df.isna().sum().to_markdown())


# Dealing with missing values

# In[10]:


# Drop 'Cabin' column due to many missing values
if 'Cabin' in df.columns:
    df.drop(['Cabin'], axis=1, inplace=True)  


# In[11]:


df['Age'].fillna(df['Age'].median(), inplace=True)


# In[12]:


# Convert categorical variables to numerical
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


# In[13]:


df.head()


# Dealing with Outliers

# In[18]:


# Define a function to identify outliers using Z-score
def identify_outliers_zscore(df, column, threshold=3):
    mean_col = np.mean(df[column])
    std_col = np.std(df[column])
    z_scores = (df[column] - mean_col) / std_col
    return df[np.abs(z_scores) > threshold]

# Identify outliers
outliers_A = identify_outliers_zscore(df, 'Pclass')
outliers_B = identify_outliers_zscore(df, 'Age')
outliers_C = identify_outliers_zscore(df, 'Fare')

print("Outliers in column A:")
print(outliers_A)
print("Outliers in column B:")
print(outliers_B)
print("Outliers in column C:")
print(outliers_C)


# In[22]:


# Optionally, remove outliers
dfc = df[~df.index.isin(outliers_A.index)]
dfc = dfc[~dfc.index.isin(outliers_B.index)]
dfc = dfc[~dfc.index.isin(outliers_C.index)]


# Feature Selection

# In[25]:


features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']


# In[26]:


# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# Scatter Plot: Age vs. Fare
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Scatter Plot: Age vs. Fare')
plt.show()

# Bar Graph: Survival Count
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()
# In[50]:


#Histogram: Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='red')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()


# In[51]:


# Pie Chart: Gender Distribution
plt.figure(figsize=(8, 8))
gender_distribution = df['Sex_male'].value_counts()
plt.pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'pink'])
plt.title('Gender Distribution')
plt.show()


# The distribution of Ages of the passengers

# In[45]:


data = pd.read_csv('C:/Users/USER/Downloads/Titanic-Dataset.csv')
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', kde=True, hue='Sex')
plt.title('Age Distribution by Gender')
plt.show()


# Survival rates based on passenger Class

# In[41]:


plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='Pclass', y='Survived', hue='Sex')
plt.title('Survival Rate by Passenger Class and Gender')
plt.show()


# Survival rates based on embarkation port

# In[42]:


plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Embarked', hue='Survived')
plt.title('Survival Count based on Embarkation Port')
plt.show()


# Survival regarding Fare and Class

# In[43]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Pclass', y='Fare', hue='Survived')
plt.ylim(0, 300)  # Limiting y-axis to 300 for better visualization
plt.title('Fare distribution by Passenger Class and Survival')
plt.show()


# Survival rates by the number of siblings/spouses aboard

# In[44]:


plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='SibSp', hue='Survived')
plt.title('Survival Count based on Number of Siblings/Spouses Aboard')
plt.show


# Data Analysis Findings
# Based on the analysis we've discussed above, here's a summary of findings for the Titanic incident:
# 
# Gender and Survival: Women had a significantly higher survival rate than men.
# 
# Passenger Class: First-class passengers had a higher survival rate, indicating socio-economic status played a role in survival chances.
# 
# Embarkation Port: The survival count varied based on the embarkation port, potentially reflecting the socio-economic distribution of passengers from these ports.
# 
# Fare and Survival: Within each passenger class, there wasn't a consistent pattern to suggest that higher fares directly led to better survival chances.
# 
# Siblings/Spouses: Those with one sibling or spouse onboard seemed to have a slightly better survival rate than those alone or with many siblings/spouses.
# 
# Age Distribution: Younger passengers (children) had a better survival rate, while the elderly had lower survival chances. Middle-aged individuals, especially males, formed the bulk of casualties.
