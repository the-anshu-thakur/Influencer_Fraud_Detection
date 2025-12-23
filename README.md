#Scenario

A popular social media platform for sharing photos and videos has received complaints about fake user accounts. These fake accounts are said to have left spam comments on genuine user posts. Management has asked us to create a machine-learning model that will help the platform distinguish real versions from fake accounts. The company would then use the model to identify fake accounts so they can be deleted from the platform.

<img width="1140" height="500" alt="image" src="https://github.com/user-attachments/assets/6c046117-e3ad-4207-8f78-c6728db00272" />

Image taken from: https://www.endnowfoundation.org/wp-content/uploads/elementor/thumbs/Detect-Fake-Profiles-on-Social-Media-p6yfct3ismgslao8tyklprwyrfd5tttfiwrd6xcjuw.jpg

**1) Goal**

The goal here is to predict whether a user account is fake or not. A problem of this nature is called a binary classification problem (binary since we have two categories). We use int numbers to specify the two categories. In the 'fake' column, a 1 represents that the account in that row is fake, while a 0 indicates a real account.

**2) Gather Data**

The data is in the social_media_train.csv file. The target vector is given by the 'fake' column. Here the modules that typically are needed for reading and exploration are imported and then read in pandas DataFrame df_train.

**3) Exploratory Data Analysis (EDA)**

Understand Data

It is necessary to familiarize ourselves with the data at the beginning so that we know later what to look for while cleaning and preparing the data.

**4) Logistic Regression**

In this notebook, we first focus on Logistic Regression which is a known algorithm for classification tasks. Please see this article%20is%20the%20sigmoid%20function,that%20the%20output%20is%200.) for more details about logistic regresion.

From regularization, we know that it can be helpful if a linear regression tries to minimize the slope values. For example, overfitting can be avoided. In contrast to the logistic regression without regularization, the features must be standardized in the model with regularization. This is due to penalty parameters (l2 by default) in logistic regression: like linear regression, logistic regression with regularization makes the prediction dependent on the scaling of the features, where at 𝐿1 (Lasso) and 𝐿2 (Ridge) large coefficients are penalized more heavily. So, in order for the coefficients to be penalized equally, we need to standardize the coefficients.

**Logistic Regression With And Without Regularization**

As with linear regression and its Ridge and Lasso versions, logistic regression also allows for regularization. Please see this article for more details about regularisations.

we can now set up a logistic regression model By default, sklearn's logistic regression algorithm already uses regularization with the regularization parameter C=1.0. If we assign an extremely large value to C, such as a 1 followed by 42 zeros (1e42), no regularization is performed. That's what we want to achieve here first.

The algorithm needs many attempts to solve the problem. The default 100 iterations are not enough. Therefore, we should also assign a relatively large number to max_iter. This parameter sets the maximum number of iterations the solvers need to converge. 10000 (1e4) should suffice here.

**5) Model Performance Evaluation**

We use test data from social_media_test.csv for initial classification performance:

model_log is better at recognizing actual fake accounts as such (recall), while model_reg has more actual fake accounts among predicted fake accounts (precision). In other words, model_log's predictions can be trusted a little less than model_reg.

**ROC (Receiver Operating Characteristic)**

LogisticRegression uses a threshold of 0.5 (50%) for determining the predicted categories. If we lower this threshold, there are more and more positive predictions and thus the recall increases. At the same time, however, the precision also decreases, because more and more of the predicted reference category cases are not fake at all.

*The question here is which classification threshold suits our specific problem well?* We can investigate this issue with the help of ROC (Receiver Operating Characteristic) curve. ROC curves are a nice way to see how any predictive classifier like logistic regression model can distinguish between the true positives and negatives.

<img width="720" height="720" alt="image" src="https://github.com/user-attachments/assets/73671067-621a-4c96-bb6b-f2a6c62db3d9" />


The ROC curve does this by plotting sensitivity, the probability of predicting a real positive will be a positive, against 1-specificity, the probability of predicting a real negative will be a positive.

<img width="720" height="720" alt="image" src="https://github.com/user-attachments/assets/0fa21621-4daf-41a4-bb9e-cd78a340eedb" />


The false positive rate is essentially a measure of how often a “false case” will occur — or, how often an actual negative instance will be classified as positive. Figure above demonstrates how some theoretical classifiers would plot on an ROC curve. The gray dotted line represents a classifier that is no better than random guessing — this will plot as a diagonal line. The purple line represents a perfect classifier — one with a true positive rate of 100% and a false positive rate of 0%. Nearly all real-world examples will fall somewhere between these two lines — not perfect, but providing more predictive power than random guessing. Typically, what we’re looking for is a classifier that maintains a high true positive rate while also having a low false positive rate — this ideal classifier would “hug” the upper left corner of Figure 1, much like the purple line in figure above. Image and text taken from: https://towardsdatascience.com/understanding-the-roc-curve-and-auc-dd4f9a192ecb

Back to our problem, how can we compare two different implemented model above? What are the false positive rates for model_log and model_reg? how the recall and the false positive rate change with the threshold. This can be investigated via ROC curve.

The roc_curve() function from sklearn.metrics calculates the values ​​of the curve. It needs the predicted probabilities. In a classification model, we always use my_model.predict() to predict the categories. Instead, if we want to know the probability of belonging to one category or another, we should use my_model.predict_proba().

The algorithm needs many attempts to solve the problem. The default 100 iterations are not enough. Therefore, we should also assign a relatively large number to max_iter. This parameter sets the maximum number of iterations the solvers need to converge. 10000 (1e4) should suffice here.

The red curve is the outcome from the model without regularization. From the recall of around 42% and from a false positive rate of approx. 19%, it hardly deviates from the ideal. So the model appears generally very good as it follows the gray ideal rather than the blue rate line. But it's not perfect.

The ROC curve of the model with regularization (model_reg) deviates a little more from the ideal gray dashed line of the model without regularization (model_log). This suggests that the logistic regression model with regularization is slightly worse in accuracy than the non-regularized model.

**ROC AUC Measure**

To create a quantifiable model performance measure from the visual impression we just used, use the area under the curve. Hence the name of the model quality measure: receiver operator characteristic area under the curve (ROC-AUC).

The ROC AUC score tells us how efficient the model is. The higher the AUC, the better the model’s performance at distinguishing between the positive and negative classes. An AUC score of 1 means the classifier can perfectly distinguish between all the Positive and the Negative class points. An AUC value of 0 shows that the classifier predicts all Negatives as Positives and vice versa.

<img width="828" height="621" alt="image" src="https://github.com/user-attachments/assets/b47c4fe3-0e18-4287-bc24-ca5ad378130f" />

Image source: https://medium.com/the-owl/evaluation-metrics-part-3-47c315e07222

Comparing the score of the model without regularization 96%, to model with regularization 77%, it confirms our visual impression above: the model with regularization follows the ideal less than the model without regularization. The regularization has thus harmed the model quality rather than helped it. Regardless of the chosen decision thresholds, we can conclude that we have overfitted our model!

**6) Find The Best Logistic Regression Model With Grid Search and ROC-AUC**

We can now combine the accumulated knowledge. We will use the ROC AUC measure to find the best logistic regression model and then use it to predict the fake status of the social media accounts. Thanks to the GridSearchCV the optimal hyperparameters of models can be found automatically. We can now turn to the pipeline containing the steps from data processing and feature engineering to prediction.

Before we can start the grid search, we still have to define the grid of hyperparameters that is to be searched:

Penalty: whether the regularization is like Ridge ('l2') or like LASSO ('l1')
C: Regularization weakness - Inverse of regularization strength

Since we want to try both regularizations of ridge regression and lasso regression, we cannot use 'lbfgs' as before. Only the 'saga' setting supports both types of regularization. Please see here for more details.

In order for the algorithm to find a good result, it has to make a relatively large number of attempts. Therefore, max_iter should be increased from the default of 100 iterations. 10000 iterations (1e4) seem to be enough.

**7) Model Evaluation With Test Data**

It is best to check whether the extraordinarily good model quality according to cross-validation is also achieved with the test data.

This value (ROC-AUC measure of 93.9%) is very good, but unfortunately slightly lower than the model figure of merit calculated using the validation data during the grid search. A discrepancy between model performance on test data and validation data is common.

**8) Prediction**

There are two Instagram accounts where the model is absolutely sure about the categorization. The predicted fake probability is either exactly 1.0 or 0.0. For the other five accounts, the model is less certain. Here the management can decide for itself which threshold value it wants to choose in order to block accounts. The classification with the default threshold of 0.5 can be seen in the 'fake_pred' column. She is a good first serve.

User accounts are now categorized into real and fake based on their characteristics. We've even exceeded management's expectations because they can now decide for themselves how secure the classification must be before an account is removed.
