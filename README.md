# case-study-of-the-Titanic-dataset
## Titanic Data Analysis
Data Science Lab Project – 2nd Semester

Guru Ghasidas Vishwavidyalaya (GGU)

### 1. Project Overview

This project performs Exploratory Data Analysis (EDA) on the Titanic dataset.
The goal is to understand passenger survival patterns based on different factors such as:

Gender

Passenger Class

Age

Survival Status

The analysis uses Python libraries to visualize and interpret the dataset.

### 2. Technologies Used

Python

Pandas

NumPy

Matplotlib

Jupyter Notebook

### 3. Libraries Used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Library Purpose
Library	Purpose
Pandas	Data manipulation and analysis
NumPy	Numerical operations
Matplotlib	Data visualization
### 4. Dataset

The project uses the Titanic dataset containing passenger information.

Main files used:

train.csv

test.csv

Dataset contains features such as:

PassengerId

Pclass

Name

Sex

Age

Fare

Embarked

Survived

### 5. Loading the Dataset
```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
```
This step loads the dataset into a Pandas DataFrame and displays the first few rows.

### 6. Gender Distribution Analysis

We count the number of male and female passengers.
```python
male_ind = len(train[train['Sex'] == 'male'])
print("No of Males in Titanic:",male_ind)

female_ind = len(train[train['Sex'] == 'female'])
print("No of Females in Titanic:",female_ind)
Visualization
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

gender = ['Male','Female']
index = [male_ind,female_ind]

ax.bar(gender,index, color=['blue','pink'])

plt.xlabel("Gender")
plt.ylabel("No of people onboarding ship")

plt.show()
```
This chart shows the distribution of male and female passengers on the Titanic.

### 7. Survival Analysis

We calculate how many passengers survived and how many died.
```python
alive = len(train[train['Survived'] == 1])
dead = len(train[train['Survived'] == 0])

print("No of people alive in Titanic:",alive)
print("No of people dead in Titanic:",dead)
Visualization
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

gender = ['alive','dead']
index = [alive,dead]

ax.bar(gender,index, color=['green','red'])

plt.xlabel("Status of people")
plt.ylabel("No of people onboarding ship")

plt.show()
```
This chart compares survivors vs non-survivors.

### 8. Survival Percentage by Gender
```python
survival_pct = train.groupby('Sex')[['Survived']].mean() * 100
print(survival_pct)
```
This calculation shows the percentage of survival for male and female passengers.

Observation:

Female survival rate was significantly higher.

Male passengers had lower survival chances.

### 9. Survival Based on Passenger Class

Passenger class played an important role in survival.

Survived Passengers by Class
```python 
plt.figure(1)

train.loc[train['Survived'] == 1, 'Pclass'].value_counts().sort_index().plot.bar()

plt.title('Survived: Ticket Class Distribution')
plt.xlabel("Class")
plt.ylabel("Number of People")
Non-Survivors by Class
plt.figure(2)

train.loc[train['Survived'] == 0, 'Pclass'].value_counts().sort_index().plot.bar()

plt.title('Did Not Survive: Ticket Class Distribution')
plt.xlabel("Class")
plt.ylabel("Number of People")

plt.show()
```
Observation:

First class passengers had a higher survival rate.

Third class passengers had the highest death rate.

### 10. Age Distribution Analysis

Age groups are created to study survival patterns.
```python 
bins = np.arange(0, 100, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
```
Age distribution helps understand which age groups had higher survival chances.

### 11. Key Findings

Important observations from the analysis:

Females had a much higher survival rate than males.

First-class passengers survived more than third-class passengers.

A large number of passengers were male.

Passenger class significantly influenced survival probability.

### 12. Conclusion

This project demonstrates how Exploratory Data Analysis (EDA) can reveal meaningful insights from data.

Using Python libraries such as Pandas and Matplotlib, we analyzed survival patterns in the Titanic dataset and discovered key factors affecting survival.

EDA is an essential step in data science and machine learning workflows.

### 13. Future Improvements

Future improvements can include:

Feature engineering

Machine learning models

Survival prediction models

Advanced visualizations
