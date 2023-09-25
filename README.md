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
