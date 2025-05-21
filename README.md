# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_dataset.csv')

print(train.head())
```
![image](https://github.com/user-attachments/assets/7924b50c-c4ad-490f-a54d-641b6a614411)

```
print(train.isnull())
```
![image](https://github.com/user-attachments/assets/f8ecfee4-f858-45bb-a875-a7c7dc2a0a00)

```
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap (Initial)")
plt.show()
```
![image](https://github.com/user-attachments/assets/9768ed77-2ecb-4f56-a103-46bc2e4a1b8f)

```
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)
plt.title("Survival Count")
plt.show()
```
![image](https://github.com/user-attachments/assets/54d447ed-0c72-464a-a1e2-4637d0c76282)

```
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r') # Corrected palette name
plt.title("Survival Count by Sex")
plt.show()
```
![image](https://github.com/user-attachments/assets/670468b3-1b98-4541-8105-71b67bac7985)

```
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')
plt.title("Survival Count by Passenger Class")
plt.show()
```
![image](https://github.com/user-attachments/assets/de632988-dbc9-4766-9d63-ce3c00c51531)

```
sns.distplot(train['Age'].dropna(), kde=False, color='darkred', bins=40)
plt.title("Age Distribution")
plt.show()
```
![image](https://github.com/user-attachments/assets/eb1f3370-74bc-49d4-9315-5b129e023a8a)

```
train['Age'].hist(bins=30, color='darkred', alpha=0.3)
plt.title("Age Histogram")
plt.show()
```
![image](https://github.com/user-attachments/assets/1e5cd68d-8adf-4939-8ae9-0a46c33d1dd3)

```
sns.countplot(x='SibSp', data=train)
plt.title("Siblings/Spouses Aboard Count")
plt.show()
```

![image](https://github.com/user-attachments/assets/ff2eeb55-d21c-4c12-ae67-0a2862f27129)

```
train['Fare'].hist(color='green', bins=40, figsize=(8, 4))
plt.title("Fare Distribution")
plt.show()
```
![image](https://github.com/user-attachments/assets/4f78b187-4375-48e3-afe9-d3df75535663)

```
import cufflinks as cf
cf.go_offline()

train['Fare'].iplot(kind='hist', bins=30, color='green', title="Fare Distribution (Interactive)")

```
![image](https://github.com/user-attachments/assets/2528b2f7-9c75-4a51-a8fc-cfdf817c1112)

```
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')
plt.title("Age Distribution by Passenger Class")
plt.show()
```

![image](https://github.com/user-attachments/assets/778dc50f-98a2-4841-b855-bbc4327fbcd8)

```
def impute_age(cols):
    Age = cols.iloc[0]
    Pclass = cols.iloc[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap (After Age Imputation)")
plt.show()

```
![image](https://github.com/user-attachments/assets/71a7eed0-c0a3-4eb8-8878-c824f2ac382d)

```
train.drop('Cabin', axis=1, inplace=True)

print("\nDataFrame Head after dropping 'Cabin':")
print(train.head())
```
![image](https://github.com/user-attachments/assets/c44f6eb5-89ee-47de-a410-aaea62d8b3e6)

```
train.dropna(inplace=True)
train.info()

```
![image](https://github.com/user-attachments/assets/715a3131-cdf1-4f53-a242-66213cd69c53)
```
print(pd.get_dummies(train['Embarked'], drop_first=True).head())
```
![image](https://github.com/user-attachments/assets/eed89f9f-e549-417d-83dd-182eb7b7ca4c)

```
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

print("\nDataFrame Head after dropping original categorical and irrelevant columns:")
print(train.head())

```
![image](https://github.com/user-attachments/assets/38c5f25c-eea6-4f85-b74c-a48b908788c7)
```
train = pd.concat([train, sex, embark], axis=1)
print("\nFinal DataFrame Head before model training:")
print(train.head())

```
![image](https://github.com/user-attachments/assets/6be9da2a-668f-4e3e-aafe-ad32374e5f8d)

```
X = train.drop('Survived', axis=1)
y = train['Survived']

print("\nX (features) head:")
print(X.head())
print("\ny (target) head:")
print(y.head())
```
![image](https://github.com/user-attachments/assets/ea0fcdd1-1d6d-47e2-80a6-9b9d8477bfb4)

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy_matrix = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(accuracy_matrix)

```
![image](https://github.com/user-attachments/assets/a6a61e6b-3506-4dc2-b255-270d5efab6f2)
```
from sklearn.metrics import accuracy_score

accuracy_score_value = accuracy_score(y_test, predictions)
print(f"\nAccuracy Score: {accuracy_score_value}")

print("\nPredictions on test set:")
print(predictions)
```

![image](https://github.com/user-attachments/assets/ab50ae8d-8064-47e5-ad6e-04b14c2b7c50)

# RESULT
Thus,Data Analyzing of the given dataset was successful.
