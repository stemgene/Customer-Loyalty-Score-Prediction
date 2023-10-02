# Customer-Loyalty-Score-Prediction

# Description

Imagine being hungry in an unfamiliar part of town and getting restaurant recommendations served up, based on your personal preferences, at just the right moment. The recommendation comes with an attached discount from your credit card provider for a local place around the corner!

C company has built partnerships with merchants in order to offer promotions or discounts to cardholders. But do these promotions work for either the consumer or the merchant? Do customers enjoy their experience? Do merchants see repeat business? Personalization is key.

C has built machine learning models to understand the most important aspects and preferences in their customers’ lifecycle, from food to shopping. But so far none of them is specifically tailored for an individual or profile. This is where you come in.

In this project, I will develop algorithms to identify and serve the most relevant opportunities to individuals, by uncovering signal in customer loyalty. I will improve customers’ lives and help C reduce unwanted campaigns, to create the right experience for customers.

## Data File Description

* train.csv - the training set
* test.csv - the test set
* sample_submission.csv - a sample submission file in the correct format - contains all `card_ids` you are expected to predict for.
* historical_transactions.csv - up to 3 months' worth of historical transactions for each `card_id`
* merchants.csv - additional information about all merchants / `merchant_ids` in the dataset.
* new_merchant_transactions.csv - two months' worth of data for each `card_id` containing ALL purchases that `card_id` made at `merchant_ids` that were not visited in the historical data.

## Evaluation

**Root Mean Squared Error (RMSE)**

Submissions are scored on the root mean squared error. RMSE is defined as:

$$RMSE=\sqrt{\frac{1}{n}\sum{(y_i-\hat{y_i})^2}}$$

where $\hat{y_i}$ is the predicted loyalty score for each `card_id`, and $y$ is the actual loyalty score assigned to a `card_id`.

# Project Process

## Preprocessing

* Partitioning discrete and continuous fields.
* Enlabel the discrete features
* Addressing the missing values with -1 or mean
* Addressing the infinite values with the maximum value of the column
* Drop duplicates

## Feature Engineering

This project contains many tables, such as transactions, merchant table. If I want to unifized all data based on `card_id`. To do so, it's important to merge data from different tables on unique `card_id`.

In this project, I'll apply two methods to integrate data.

### Generate new features based on sum of values

The orignal table

|`card_id`| A | B | C | D |
|  --    | -- | -- | -- | -- |
| --     | Categorical | Categorical | Numeric | Numeric |
| 1 | 1 | 2 | 4 | 7 |
| 2 | 2 | 1 | 5 | 5 |
| 1 | 1 | 2 | 1 | 4 |
| 3 | 2 | 2 | 5 | 8 |

The new table generated new features

|`card_id`| A1 & C | A2 & C | A1 & D | A2 & D | B1 & C | B2 & C | B1 & D | B2 & D | 
|  --    | -- | -- | -- | -- | -- | -- | -- | -- | 
| 1 | 5 | 0 | 11 | 0 | 0 |  5 | 0 | 11 |
| 2 | 0 | 5 | 0 | 5 | 5 | 0 | 5 | 0 |
| 3 | 0 | 5 | 0 | 8 | 0 | 5 | 0 | 8 |

This method will cause enormous new columns and zero values.

### Generate new features baed on statistic metrics

The new table generated new features

|`card_id`| A_sum | A_mean | A_max | A_var | A_unique | 
|  --    | -- | -- | -- | -- | -- |
| 1 | 2 | 1 | 1 | 0 | 2 |  
| 2 | 2 | 2 | 2 | 0 | 2 | 
| 3 | 2 | 5 | 2 | 0 | 2 | 

This method won't cause so many new columns or zero values.

## Training models

During this project, I've implemented serval approaches, from basic to advanced, to improve the metrics.

### Base line: Random Forest + Grid search

The RMSE on the test set is 3.65455

### Random Forest + 5 Fold Cross Validation

| Approach | RMSE |
| -- | -- |
| Base line | 3.65455 |
| RF + CV | 3.65173 |

### Wrapper Feature Selection + LightGBM + TPE optimize

#### Feature selection

Select top 300 features according to correlation with the target to train

| Approach | RMSE |
| -- | -- |
| Base line | 3.65455 |
| RF + CV | 3.65173 |
| LightGBM | 3.69732 |
| LightGBM + CV| 3.64403 |

#### NLP Feature Engineering + XGBoost + Bayes Optimization

I found there were many columns related to ID, such as `merchant_id`, `merchant_category_id`, `state_id`, `subsector_id`, `city_id`, these values inflect customer's behavior. For example, for a customer A, if a certain merchant id happens a lot in his transaction record, it means A likes this merchant. Furthermore, if this merchant record in many customers', it means this merchant is popular, and it also means customer A is similar to other customers, if not, A has a special like. 

In order to mining this information, I will apply CountVector and TF-IDF. Specificly, CountVector can extract merchant information of a customer, and TF-IDF can extract if many customers like one product at the same time.

If we apply NLP approaches, there is an issue we should consider which is there are too many new features and most of them are sparse. So I'll introduce associated method `sparse` from `scipy`.

| Approach | RMSE |
| -- | -- |
| Base line | 3.65455 |
| RF + CV | 3.65173 |
| LightGBM | 3.69732 |
| LightGBM + CV| 3.64403 |
| XGBoost | 3.62832 |

## Ensemble training

### Voting

* Voting with average

Average three results from models (LGBM, RF, XGBoost)

| Approach | RMSE |
| -- | -- |
| Base line | 3.65455 |
| RF + CV | 3.65173 |
| LightGBM | 3.69732 |
| LightGBM + CV| 3.64403 |
| XGBoost | 3.62832 |
| Average Voting | 3.6365 |

Only simple average cannot have better results.

* Voting with weighted average

| Approach | RMSE |
| -- | -- |
| Base line | 3.65455 |
| RF + CV | 3.65173 |
| LightGBM | 3.69732 |
| LightGBM + CV| 3.64403 |
| XGBoost | 3.62832 |
| Average Voting | 3.6365 |
| weighted Voting | 3.633307 |

### Stacking

| Approach | RMSE |
| -- | -- |
| Base line | 3.65455 |
| RF + CV | 3.65173 |
| LightGBM | 3.69732 |
| LightGBM + CV| 3.64403 |
| XGBoost | 3.62832 |
| Average Voting | 3.6365 |
| weighted Voting | 3.633307 |
| Stacking | 3.62798 |

## Advanced optimization

Generally speaking, there are two apporaches to train a better model. The first one is to improve the data quality by feature engineering. The other one is to improve the quality of model. Comparing with these two features, the upper limit of the model accuracy depends on feature engineering, and the building model will approach this upper limit. Therefore, during the optimization process, we need both these two method in order to get a better result.

### Feature engineering

I've already made some feature engineering tasks, e.g. aggregate data by `card_id`, generate more features, feature selection, and extract information by NLP. There are still improve space for feature engineering.

#### Customer behavior

During the previous process, I ignored an important customer behavior feature, i.e. the time feature. We could create new features based on this information, such as time difference between the first trade and the latest trade, average time interval between a customer transaction, etc. 

In a word, the customer behavior features are important to extract information and make better prediction results efficiently. Additionally, the closer a customer behavior to the current time point, the more valuable. So I focus on the customer behavior of the latest two months.

#### Second-order cross derivation

I've generated new features according cross derivation, such as summarize two features. This time I'll apply higher order to generate new features. Of course, the higher order will cause more sparse matrix, so I'll only multiply features related to customer behavior.

#### Abnormal values

I've addressed the abnormal values and removed the values which less than -30. However, according to the data description, these abnormal values are manipulated by the company. It represents a certain type of customer. Therefore I will not remove them, but consider them as a group of customers.

### Replace the Random Forest with the CatBoost

| Approach | RMSE |
| -- | -- |
| Base line | 3.65455 |
| RF + CV | 3.65173 |
| LightGBM | 3.69732 |
| LightGBM + CV| 3.64403 |
| XGBoost | 3.62832 |
| Average Voting | 3.6365 |
| weighted Voting | 3.633307 |
| Stacking | 3.62798 |
| LightGBM | 3.61177 |
| XGboost | 3.61048 |
| CatBoost | 3.61154 | 

### Ensemble + Voting + stacking

| Approach | RMSE |
| -- | -- |
| Base line | 3.65455 |
| RF + CV | 3.65173 |
| LightGBM | 3.69732 |
| LightGBM + CV| 3.64403 |
| XGBoost | 3.62832 |
| Average Voting | 3.6365 |
| weighted Voting | 3.633307 |
| Stacking | 3.62798 |
| LightGBM | 3.61177 |
| XGboost | 3.61048 |
| CatBoost | 3.61154 | 
| Voting | 3.60640 |
| Stacking | 3.60683 | 