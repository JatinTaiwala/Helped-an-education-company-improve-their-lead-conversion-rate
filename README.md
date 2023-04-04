# Lead-scoring-case-study-Logistic_regression

## Overview
This project was undertaken for X Education, an online education company that sells courses to industry professionals. The goal was to build a logistic regression model to assign a lead score between 0 and 100 to each lead, indicating their likelihood to convert into a paying customer. The objective was to identify the most promising leads (i.e., hot leads), allowing the sales team to focus on communicating with potential customers and increasing the lead conversion rate to 80%.

## Data
The dataset provided had around 9000 data points with various attributes such as Lead Source, Total Time Spent on Website, Total Visits, Last Activity, etc. The target variable was 'Converted,' which indicates whether a past lead was converted or not, where 1 means it was converted and 0 means it wasn't converted.

## Approach
The dataset was cleaned by handling the null values and levels present in the categorical variables. The data was then split into train and test datasets. A logistic regression model was built on the train dataset, with feature selection using Recursive Feature Elimination. The model was then tuned with hyperparameters to achieve a balanced accuracy of 81% on the train dataset.

## Results
The model was evaluated on the test dataset, achieving an accuracy of 80.4%, sensitivity of 80.4%, and specificity of 80.5%. The results show that the model can predict the conversion rate well, with a lead score equal to or greater than 85 indicating 'Hot Leads' that the sales team should focus on contacting.

## Conclusion
The logistic regression model built in this project can help X Education to identify 'Hot Leads' and increase the lead conversion rate to 80%. The code and methodology used can be easily replicated and applied to other datasets to predict the likelihood of an outcome.
