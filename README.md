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

Further, certain types of non-consumables, i.e. those in the categories *Health and Hygiene*, *Household* and *Others* are either *Low Fat* or *Regular* according to the data. Clearly, this makes no sense. Hence, we introduce an new fat level *None* for non-consumables.

#### Item weights
Checking for missing values, we see that 2439 entries are missing in the category *Item_Weight*. Let's explore those weights a little.
Looking at a boxplot of the weights grouped by the outlet identifier we see that *OUT019* and *OUT027* have not reported any weight data:
![alt text](https://github.com/MichaelPluemacher/Big-Mart-Sales/blob/master/Graphs/Weight_vs_Outlet_1.png)
Fortunately, all the wares on offer in those stores are also sold elsewhere. Assuming that each *Item_Identifier* actually identifies a unique item we can impute the missing weights by getting those weights from other stores. Fortunately, this successfully filled all the mising values:
![alt text](https://github.com/MichaelPluemacher/Big-Mart-Sales/blob/master/Graphs/Weight_vs_Outlet_2.png)
Looking at those plots one notices that all the medians, boxes and whiskers are identical to each other. Have those practice data been faked by any chance?

#### Year a shop has been operating
The sales data are for the year 2013. Also, the data contain the year in which each shop was established. For convenience, we replace that value by the number of years each shop has been in existence before 2013.

#### Item list price
Looking at the density of the list price of items (*Item_MRP*),
![alt text](https://github.com/MichaelPluemacher/Big-Mart-Sales/blob/master/Graphs/Item_MRP_Density.png)
we clearly see that there are four different price categories. To differentiate between them we introduced a new factor with four price levels: *Low*, *Medium*, *High*. and *Very_High*.

#### Outlet size
Some entries in the category *Outlet_Size* are empty. To tackle that problem, let's explore sales in various outlets. Counting how many sales where reported by each outlet,

| Outlet_ID | number of sales |
|-----------|----------------:|
| OUT010 |  925 |
| OUT013 | 1553 |
| OUT017 | 1543 |
| OUT018 | 1546 |
| OUT019 |  880 |
| OUT027 | 1559 |
| OUT035 | 1550 |
| OUT045 | 1548 |
| OUT046 | 1550 |
| OUT049 | 1550 |

we see that the two grocery stores *OUT010* and *OUT019* have reported far fewer sales than the supermarkets. This is neatly illustrated by a boxplot:
![alt text](https://github.com/MichaelPluemacher/Big-Mart-Sales/blob/master/Graphs/Sales_vs_OutletID.png)
Grouping sales by the type of outlet and the years it has existed,
![alt text](https://github.com/MichaelPluemacher/Big-Mart-Sales/blob/master/Graphs/Sales_vs_OutletType_1.png)
or the type of outlet and its size,
![alt text](https://github.com/MichaelPluemacher/Big-Mart-Sales/blob/master/Graphs/Sales_vs_OutletType_2.png)
we see that there is a clear distinction in sales figures between grocery stores and supermarkets. This is confirmed if we look at sales figures across various item categories:
![alt text](https://github.com/MichaelPluemacher/Big-Mart-Sales/blob/master/Graphs/Sales_vs_ItemType_2.png)

However, the various types of supermarkets cannot be distinguished that easily. This is probably due to other factors, e.g. their location, how long they have been in operation, how well they are managed, etc. In particular, sales in the one Type 2 supermarket in the data are somewhat low. This may be due to the fact that it is still fairly new, having been founded four years ago.

Coming back to the lower sales figures for grocery stores, from the description of the data it is not immediately clear why that is so. A reasonable assumption is that grocery stores are much smaller than supermarkets and therefore only have a reduced selection of wares on offer. This is confirmed by a simple count of item identifiers in each outlet.

The missing values in the outlet size category concern one grocery store and two type 1 supermarkets. From what we have seen above, the grocery store clearly falls in the category *Small*. From the sales figures the type 1 supermarkets could be either *Small* or *Medium*. Since type 1 supermarkets are most often classified as small, we replace those missing size levels by *Small*. 

#### Item visibilities





