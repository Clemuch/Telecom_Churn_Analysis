# Telecom_Churn Analysis

# Table of Contents
[Project Overview](Project Overview)

## Project Overview
This dataset aims to analyze customer behavior and predict churn (whether a customer will discontinue services). Here's a detailed overview:

1. Dataset Structure
- Rows (observations): 7043
- Columns (features): 21
- Purpose: Understand customer demographics, account information, and service usage to predict churn.

2. Data Characteristics
- Categorical Variables: Many columns such as gender, Partner, Dependents, Contract, etc., are categorical, ideal for classification analysis.
- Numerical Variables: Columns like tenure, MonthlyCharges, and TotalCharges provide continuous data for analysis.
- Target Variable (Churn): Binary classification target for predicting customer churn.

3. Data Insights
- Churn Analysis: The dataset contains information on customers who churned and those who did not. Understanding their characteristics can help prevent future churn.
- Service Utilization: Features like OnlineSecurity, StreamingTV, and TechSupport provide insights into customer preferences.
- Payment Patterns: Columns such as PaymentMethod and PaperlessBilling show how payment behavior correlates with churn.

Will conduct data cleaning and perform exploratory data analysis. The dataset is gotten from kaggle.

## Data Source

The telecom dataset is gotten from [kaggle](https://www.kaggle.com/code/bandiatindra/telecom-churn-prediction/input).

## Tools
1. Jupyter Notebook
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Visualization
   
## Framework
1. Pandas
2. Matplotlib
3. Seaborn
4. Sklearn
5. Scipy

## Data Cleaning and Preparation
During the data preparation phase, we performed the following task;
1. Data loading using pandas
2. Handling missing values
3. Checking for duplicates
4. Droping columns not needed (CustomerID)
5. Handling inconsistent data type

## Exploratory Data Analysis (EDA)

### 1. correlation 
Converting all categorical data to dummy using pandas frame work 'pd.get_dummies()'.


```
# Python Code
# Let's convert all the categorical variables into dummy variables

data_dummies = pd.get_dummies(data)

# Get Correlation of "Churn" with other variables:

plt.figure(figsize=(15,8))
data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
plt.show()

```
**Output**
![correlation](https://github.com/user-attachments/assets/72bb8206-6e0f-436b-ab51-2f6a98a8e9f5)

**Observation**

**Positive Correlations:**

Features like Contract_Month-to-month, OnlineSecurity_No, and TechSupport_No have strong positive correlations with Churn.

**Insights**: Customers with these attributes are more likely to churn. For example:
Customers on a month-to-month contract might find it easier to leave compared to those with longer commitments.
Lack of online security or tech support might make customers feel less satisfied.

**Negative Correlations:**

Features like tenure, Contract_Two_year, and InternetService_Fiber_optic have strong negative correlations with Churn.

**Insights**: Customers with higher tenure or longer contracts are less likely to churn. Similarly, those who use fiber-optic internet may have a better experience, reducing churn.

**Recommendations**
- Offering incentives to month-to-month customers to switch to longer contracts.
- Providing or enhancing online security and tech support services.

### Count and percentage of customers who churned

``` Python
# Count plots of customers who churned
ax = sns.countplot(x = 'Churn', data=data)
ax.bar_label(ax.containers[0])
plt.xticks(ticks=[0,1], labels=['No', 'Yes'])
plt.title('Count of Customers who churned')
plt.show()

# Let's get the percentage of custormers who churned
churn_count = data['Churn'].value_counts()
plt.pie(churn_count, labels=['No','Yes'], autopct='%1.1f%%', shadow=True, colors=['Brown', 'Teal'])
plt.title('Percentage of Customers who churned')
plt.show()
```
Output

![customer churn](https://github.com/user-attachments/assets/6e292ade-bada-4410-adbd-052e6584874d)

![percent customers who churned](https://github.com/user-attachments/assets/1c773ec9-2434-4c9c-921e-a381bca942ee)

**Insights**
- Bar chart: The number of customers who did not churn outweights the number of customers who churned. 
- Pie chart: 26% of customer churned while 73% remained loyal.

### Churn by SeniorCitizen 
```Python Code
# Let's get the percentage of Senior Citizens
churn_percent = data['SeniorCitizen'].value_counts()
plt.pie(churn_percent, labels=['Non Senior Citizen', 'Senior Citizen'], 
        autopct='%1.1f%%', shadow=True, 
        colors=['Brown', 'Teal'])
plt.title('Percentage of Senior Citizens')
plt.show()

# Count plot of customers who churned based on their seniorcitizenship
ax = sns.countplot(x='SeniorCitizen', data=data, hue='Churn')
ax.bar_label(ax.containers[0])
plt.title('Count of Customers who churned based on their SeniorCitizenship')
plt.xticks(ticks=[0,1], labels=['Non-SeniorCitizen', 'SeniorCitizen'])
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.show()

```
**Output**

![Percent of senior citizens](https://github.com/user-attachments/assets/610d3ed7-d22d-4622-a2e8-03380c08569b)

**Insights**
Only 16% of the customers are senior citizens. 83% of customers in the datasets are young.

![Churned senior](https://github.com/user-attachments/assets/d90525a9-08bd-45b8-91b7-d134c7c2b3c1)

**Insights from the plot:**

- From the plot, it is evident that a higher number of senior citizens did not churn compared to those who did.
- This indicates that senior citizens are less likely to churn compared to other age groups.

### Customers who churned based on contract
```
# Count plot of customers who churned based on their contract
ax = sns.countplot(x='Contract', data=data, hue='Churn')
ax.bar_label(ax.containers[0])
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.title('Count of Customers who churned based on their Contract')
plt.show()
```
**Output**

![Churned by Contract](https://github.com/user-attachments/assets/ae4191b4-74df-4d26-a7a4-4c6a1b60ef58)

**Insights from the plot:**
- Customers with shorter contract types tend to have a higher churn rate.
- The company should focus on extending contract durations, as longer contracts are associated with lower churn rates.


### Gender Distribution

```
import matplotlib.ticker as mtick

colors = ['#4D3425','#E4512B']
ax = (data['gender'].value_counts()*100.0 /len(data)).plot(kind='bar',
     stacked = True,
            rot = 0,
                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers')
ax.set_xlabel('Gender')
ax.set_ylabel('% Customers')
ax.set_title('Gender Distribution')

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x()+.15, i.get_height()-3.5, \
            str(round((i.get_height()/total), 1))+'%',
            fontsize=12,
            color='white',
           weight = 'bold')

```
**Output**

![Gender Dist](https://github.com/user-attachments/assets/12179afa-08fa-4d66-a7aa-a934a80b5e54)


**Insights from the plot:**
- About half of the customers are male while the other half are female

``` Python
# Calculate the churn rate by gender
gender_churn_rate = data.groupby('gender')['Churn'].mean() * 100
print(gender_churn_rate)
ax = sns.countplot(x='gender', data=data, hue='Churn')
ax.bar_label(ax.containers[0])
plt.title('Gender Churn Rate')
plt.show()

```
**Output**

![Gender Churn](https://github.com/user-attachments/assets/63f252e4-4301-48bf-b394-8e1601f2f7f3)

**Insights from the plot:**
- There is only a slight difference in the churn rate between genders.
- Both genders exhibit the same likelihood of churning.

### Subplots

``` Python
# Count plot by the following services
services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
for i, item in enumerate(services):
    row, col = divmod(i, 3)
    ax = sns.countplot(x=item, hue='Churn', data=data, ax=axes[row, col])
    ax.set_title(item)
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])

plt.tight_layout()
plt.show()
```

**Output**

![Subplots](https://github.com/user-attachments/assets/bd4264a6-3bc1-40d6-a4f9-d87757a1b9e7)


**Analysis of each subplot:**

#### 1. **Phone Service**:
   - Most customers have phone service, with a higher count of customers who did not churn compared to those who did.
   - Customers without phone service also show fewer churn instances.

#### 2. **Multiple Lines**:
   - The churn rates are similar for customers with or without multiple lines.
   - Customers with no phone service have the lowest churn numbers (similar to the "Phone Service" plot).

#### 3. **Internet Service**:
   - Customers using fiber-optic internet have a much higher churn rate compared to DSL users.
   - Those without internet service have the lowest churn numbers.

#### 4. **Online Security**:
   - Customers without online security have a significantly higher churn rate compared to those with online security.
   - Customers with no internet service show no significant churn impact.

#### 5. **Online Backup**:
   - Similar to online security, customers without online backup have higher churn rates.
   - Those with online backup services have lower churn rates.

#### 6. **Device Protection**:
   - Customers without device protection are more likely to churn compared to those with device protection.

### 7. **Tech Support**:
   - Customers without tech support have significantly higher churn rates compared to those who have tech support.

### 8. **Streaming TV**:
   - Customers who subscribe to streaming TV services show slightly lower churn rates compared to those without.

#### 9. **Streaming Movies**:
   - A similar pattern is observed with streaming movies; customers without these services have higher churn rates.

### Key Insights:
1. **Internet Services**:
   - Fiber-optic users show a higher churn rate. The company might need to investigate service quality or pricing for fiber-optic customers.
   - Customers without online security, backup, or tech support are more likely to churn. Providing bundled services could reduce churn.

2. **Phone Services**:
   - Churn rates are relatively consistent for customers with or without multiple lines.
   - Customers without phone services have the lowest churn, but they are also fewer in number.

3. **Streaming Services**:
   - Customers using streaming services (TV and movies) churn less, indicating that offering entertainment options might improve retention.

### Churn by Tenure

``` Python
# Count plot of customers who churned based on their tenure
plt.figure(figsize=(10, 6))
ax = sns.histplot(data=data, x='tenure', hue='Churn', multiple='stack', bins=30)
ax.set_title('Churn by Tenure')
plt.xlabel('Tenure (months)')
plt.ylabel('Number of Customers')
plt.show()
```
**Output**

![Churn by Tenure](https://github.com/user-attachments/assets/0e1bae49-00e6-4157-910f-a0be006964eb)

**Insights from the plot:**
- Customers with shorter tenure (less than 10 months) have a higher churn rate.
- As tenure increases, the number of customers who churn decreases.
- Customers with longer tenure (more than 60 months) are less likely to churn.
- This indicates that customers who stay longer with the company are more loyal and less likely to leave.

### Relationship between monthly charge and total charge

``` Python
# Scatter plot of MonthlyCharges and TotalCharges
plt.figure(figsize=(10, 6))
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', data=data, hue='Churn', palette=colors)
plt.title('Scatter plot of Monthly Charges vs Total Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
plt.show()
```
**Output**

![Relationship](https://github.com/user-attachments/assets/fa444ddf-88dc-44a9-9c84-ab0d867eadb2)

**Insights from plot:**

- As monthly charges increases total charges increases.

### Churned Customers by Payment Method

``` Python
# Count plot of churn customers by payment method
plt.figure(figsize=(8,4))
ax = sns.countplot(x='PaymentMethod', data=data, hue='Churn')
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
plt.title('Churn by Payment Method')
plt.xticks(rotation=45)
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.show()
```
**Output**

![Pay method](https://github.com/user-attachments/assets/307b6240-e679-441c-ae53-4cb3fc26425d)


**Insights from the plot:**

- Electronic Check: Customers paying through electronic checks are more likely to churn, suggesting potential dissatisfaction with this payment method or its associated processes.

- Automatic Payments (Bank Transfer and Credit Card): Automatic payment methods are associated with the lowest churn rates, possibly due to convenience and reliability.

- Mailed Check: While not as effective as automatic payment methods, churn rates for mailed checks are significantly lower than electronic checks.

**Recommendations:**

- Promote Automatic Payment Methods: Encourage customers to switch to automatic payment methods like bank transfers or credit cards, as these are linked to lower churn rates.
- Investigate Electronic Check Issues: Analyze why electronic check users have higher churn rates and address any pain points, such as complexity, delays, or fees.
- Incentivize Retention: Offer loyalty rewards or discounts for customers using high-risk payment methods like electronic checks to reduce churn.


## Predictive Modeling

### Logistic Regression

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# We will use the data frame where we had created dummy variables
X = data_dummies.drop(columns='Churn')
y = data_dummies['Churn']

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Running logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)

from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))
```

**Output**

0.8075829383886256

### Random Forest

```
from sklearn.ensemble import RandomForestClassifier

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Initialize the RandomForestClassifier with corrected max_features parameter
model_rf = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1,
                                  random_state=50, max_features="sqrt",
                                  max_leaf_nodes=30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print(metrics.accuracy_score(y_test, prediction_test))
```
**Output**

0.8088130774697939

### Support Vecor Machine (SVM)

```Python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

from sklearn.svm import SVC

model.svm = SVC(kernel='linear') 
model.svm.fit(X_train,y_train)
preds = model.svm.predict(X_test)
metrics.accuracy_score(y_test, preds)
```
**Output**

0.820184790334044

### ADA Boost

```Python
# AdaBoost Algorithm
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
# n_estimators = 50 (default value) 
# base_estimator = DecisionTreeClassifier (default value)
model.fit(X_train,y_train)
preds = model.predict(X_test)
metrics.accuracy_score(y_test, preds)
```

**Output**

0.8159203980099502

## XG Boost

```Python
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
metrics.accuracy_score(y_test, preds)
```
**Output**



## Conclusions, General Insights and Recommendation

The analysis of the telecom churn dataset has provided valuable insights into the factors influencing customer churn. Here are the key takeaways:


1. **Churn Rate**:
    - Approximately 26% of customers have churned, while 74% have remained loyal.

2. **Demographics**:
    - Only 16% of the customers are senior citizens, and they are less likely to churn compared to younger customers.
    - Gender does not significantly impact churn rates, as both males and females exhibit similar churn behavior.

3. **Contract Type**:
    - Customers with month-to-month contracts have a higher churn rate compared to those with one-year or two-year contracts.
    - Longer contracts are associated with lower churn rates.

4. **Service Usage**:
    - Customers without online security, online backup, or tech support are more likely to churn.
    - Fiber-optic internet users have a higher churn rate compared to DSL users.
    - Customers using streaming services (TV and movies) tend to churn less.

5. **Payment Methods**:
    - Customers paying through electronic checks are more likely to churn.
    - Automatic payment methods (bank transfer and credit card) are associated with lower churn rates.

6. **Billing**:
    - Paperless billing has a slight impact on churn, with customers using paperless billing being slightly more likely to churn.

### Recommendations

1. **Incentivize Long-term Contracts**:
    - Offer attractive incentives for customers to switch from month-to-month contracts to longer-term contracts, such as reduced rates or added services.

2. **Enhance Service Quality**:
    - Investigate and address issues related to fiber-optic service quality or pricing to reduce churn among these customers.
    - Improve technical support services and consider offering them as part of the service package or at a discounted rate.
    - Include online security and backup services in bundled packages or offer them at reduced costs to increase uptake and satisfaction.

3. **Promote Convenient Payment Methods**:
    - Encourage the use of more convenient payment methods like automatic bank transfers or credit cards instead of electronic checks.

4. **Targeted Campaigns for Senior Citizens**:
    - Develop marketing and service initiatives specifically tailored to senior citizens, addressing their unique needs and preferences.

5. **Optimize Streaming Services**:
    - Ensure that streaming TV and movie services are reliable, offer a wide range of content, and meet customer expectations to reduce churn.

6. **Implement Loyalty Programs**:
    - Offer discounts, rewards, or special offers to customers with high total charges to encourage them to stay.

7. **Promote Paperless Billing**:
    - While it has a slight impact, promoting paperless billing can still be beneficial for cost-saving and environmental reasons, which can improve overall customer satisfaction.

By focusing on these key areas, you can develop targeted strategies to retain customers and reduce churn effectively.

