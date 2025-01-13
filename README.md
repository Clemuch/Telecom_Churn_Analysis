# Telecom_Churn Analysis

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

## Predictive Modeling

