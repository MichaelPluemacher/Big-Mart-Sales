# Big-Mart-Sales

## Introduction

This is my contribution to the Big Mart Sales prediction competition at:

http://datahack.analyticsvidhya.com/contest/practice-problem-bigmart-sales-prediction

Given sales data for 1559 products across 10 stores of the Big Mart chain in various cities the task is to build a model to predict sales for each particular product in different stores.

The train and test data, which can be found at the link given above, contain the following variables:

| Variable | Description |
|----------|------------:|
| Item_Identifier | Unique product ID|
| Item_Weight | Weight of product |
| Item_Fat_Content | Whether the product is low fat or not|
| Item_Visibility | % of total display area in store allocated to this product |
| Item_Type | Category to which product belongs|
| Item_MRP | Maximum Retail Price (list price) of product|
| Outlet_Identifier | Unique store ID |
| Outlet_Establishment_Year | Year in which store was established|
| Outlet_Size | Size of the store|
| Outlet_Location_Type | Type of city in which store is located|
| Outlet_Type | Grocery store or some sort of supermarket|
| Item_Outlet_Sales| Sales of product in particular store. This is the outcome variable to be predicted.|

## Data Exploration and Preparation
For no particular reason I decided to tackle this challenge in R. A first analysis of the data, treatment of missing values and outliers, some feature engineering, and, finally, ordering of the predictor variables by their importance in fitting a random forest model was performed with the script `AnalyzeAndClean.R`.

#### The fat content
The original data contain five different levels for the fat content: *LF, low fat, Low Fat, reg,* and *Regular*. Clearly, *LF, low fat,* and *Low Fat* are the same, as are *reg* and *Regular*. Hence, we replace *LF* and *low fat* by *Low Fat* and *reg* by *Regular*.




	


