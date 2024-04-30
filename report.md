# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Tanu Tomar

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
In the spirit of machine learning's guiding principle, "garbage in, garbage out," I encountered some interesting insights while attempting to submit predictions.Machine learning models rely heavily on the quality of the data they are trained on. In this specific case, the AutoGluon library reported a root mean squared error (RMSE) of -0.68, which initially indicated a good performing model. However, when I submitted predictions without any data preprocessing, the Kaggle competition score turned out to be quite low. This highlighted the importance of data preparation in achieving accurate predictions.
The low Kaggle score suggested that the model, despite its decent RMSE, wasn't able to understand the training data correctly. 

* Temporal Features: Enhance the model's understanding of time-related dependencies by creating new features such as 'hour'. Time-related features can help capture cyclical trends in bike demand.
* Categorical Feature Engineering: Weather, Temperature, and Humidity: Recognizing the impact of weather conditions on bike demand, I found it necessary to develop categorical variables for weather-related features. By transforming temperature, wind, and humidity into categorical data (like 'high', 'medium', 'low' categories), the model can more readily utilize these variables to predict demand variations under different weather conditions.
* Feature Encoding: The categorical features derived from weather and time data as well as season attrinute required appropriate encoding to ensure they were suitably formatted for machine learning analysis. One-hot encoding seemed a strategic choice to prevent the model from misinterpreting the ordinal nature of these features.
* Hyperparameter Tuning: Given the model's initial success yet subpar competition performance, it became evident that fine-tuning the hyperparameters could potentially enhance model accuracy. 

### What was the top ranked model that performed?
![model_test_score.png](img/model_test_score.png)
Based on the results presented in the leaderboard, the top-ranked model that performed best is the WeightedEnsemble_L3, with a root mean squared error (RMSE) of -63.791227.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
TODO: Add your explanation

### How much better did your model preform after adding additional features and why do you think that is?
TODO: Add your explanation

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
TODO: Add your explanation

### If you were given more time with this dataset, where do you think you would spend more time?
TODO: Add your explanation

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|?|?|?|?|
|add_features|?|?|?|?|
|hpo|?|?|?|?|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
TODO: Add your explanation
