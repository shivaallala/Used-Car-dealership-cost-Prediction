# What Drives the price of a car

### Project Overview:

In this project, we aim to analyze a dataset sourced from Kaggle containing information on 426,000 cars. The objective is to identify the key factors that influence the pricing of used cars. By leveraging machine learning techniques, we seek to develop a predictive model that accurately estimates the price of a used car based on various features. Ultimately, we aim to provide actionable insights and recommendations to consumers on what aspects they should prioritize when purchasing a used car.

# CRISP-DM Framework:

We will follow the CRISP-DM framework, a widely-used process in the industry for data projects:

#### Business Understanding

- **Objective:** Develop a predictive model to identify factors influencing used car prices.
- Collect, clean, and prepare a large dataset of used cars for analysis.

#### Data Exploration

- Explore dataset features, determining their nature (categorical vs. numerical, discrete vs. continuous).
- Examine unique values to assess feature relevance and transformation needs.

#### Data Preparation

- Remove missing data, outliers, and duplicate records to ensure data quality.
- Perform necessary transformations, e.g., logarithmic transformation of the target variable.
- Finalize feature set and target variable for regression modeling.
- Split dataset into training and testing sets.

#### Regression Modeling

- Build pipeline for feature transformations and execute linear regression models.
- Compare RIDGE and LASSO linear regression models.
- Optimize regularization hyperparameters using grid search.
- Evaluate model performance using appropriate metrics.

#### Evaluation

- Conduct coefficient importance and residual analysis to assess model performance and feature significance.

#### Deployment

- Share key insights and business recommendations based on the analysis.
- Outline next steps for further refinement and improvement of the predictive model.

Our analysis aims to uncover insights into the factors driving used car prices, providing valuable guidance to consumers. We will highlight significant features and their impact on pricing, offering actionable recommendations for informed decision-making.


### Data Understand and Exploration:

![Cars Info](C:\Users\shiva\OneDrive\Desktop\UC Berkeley\Used-Car-dealership-cost-Prediction\Images\Cars_info.png)


![Numeric Features pairplot]()

![Categorical Feature realation to price]()

This dataset analysis provides insights into various attributes of used cars, including region, manufacturer, model, condition, cylinders, fuel type, title status, transmission, drive, size, type, paint color, and state. These insights offer valuable information for understanding market trends, consumer preferences, and regional variations in the used car market.

##### Data Description

1. **Region**
    - The dataset contains entries from various regions.
    - Columbus has the highest count (3608 entries), followed by Jacksonville (3562 entries).
    - Some regions have very few entries, such as Southwest MS (14 entries), Kansas City (11 entries), and Fort Smith, AR (9 entries).

2. **Manufacturer**
    - Ford has the highest count with 70985 entries, followed by Chevrolet (55064 entries) and Toyota (34202 entries).
    - Some manufacturers have relatively low counts, such as Ferrari (95 entries), Datsun (63 entries), and Morgan (3 entries).

![Manufacturer by Average price]()

3. **Model**
    - The dataset includes a wide variety of car models.
    - Popular models like F-150, Silverado 1500, and Camry have high counts.
    - Many models have very low counts, indicating a diverse range of vehicles in the dataset.

4. **Condition**
    - The majority of cars are listed in "good" or "excellent" condition.
    - There are fewer listings in "like new" or "new" condition.

5. **Cylinders**
    - Most cars have 6 cylinders, followed by 4 and 8 cylinders.
    - Other cylinder counts are much less common.

6. **Fuel**
    - Gasoline (gas) is the most common fuel type, followed by diesel and hybrid.
    - Electric-powered cars have the lowest count.

7. **Title Status**
    - The majority of cars have a "clean" title status.
    - Some cars have "rebuilt" or "salvage" titles.

8. **Transmission**
    - Automatic transmission is the most common type.
    - Manual transmission is less common.

9. **Drive**
    - Cars with 4-wheel drive (4wd) have the highest count.
    - Front-wheel drive (fwd) and rear-wheel drive (rwd) are also common.

10. **Size**
    - Full-size vehicles are the most common, followed by mid-size and compact.
    - Sub-compact vehicles have relatively fewer entries.

11. **Type**
    - Sedans are the most common type of vehicle, followed by SUVs and pickups.
    - Less common types include buses and off-road vehicles.

12. **Paint Color**
    - White, black, and silver are the most popular paint colors.
    - Less common colors include purple and orange.

13. **State**
    - California has the highest count of car listings, followed by Florida and Texas.
    - Some states have relatively few listings.
    -Imbalanced Distribution: Attributes with highly imbalanced distributions, where one or a few categories dominate the majority of the data, may not contribute much to the model's performance. For example, the "state" attribute has highly imbalanced distributions, with some states having significantly more observations than others. Region attribute plays a similar role as it poses insigificant relevance to the analysis or prediction. Also evaluating 'region' attribute's cardinality expresses the high degree of unique values, approximately 404 unique values. Based on this rationale, it is deamed appropreiate to remove 'region' as well.

    ![State by Average Price]()

These insights provide valuable information about the distribution of car attributes in the dataset. Further analysis can be conducted to explore market trends, consumer preferences, and regional variations.


 ##### Extreme outliers for price and odometer

 ![num features stats]()

 ![price and odometer boxplot]()

 The dataset's average car price stands at $75,199.03, with a maximum recorded price reaching a staggering $3,736,928,711. Similarly, the average odometer reading for cars in the dataset is approximately 98,043.33 miles, with the maximum odometer reading reported at 10,000,000 miles. It's noteworthy that both price and odometer readings exhibit outliers, suggesting the presence of extreme values that may significantly skew the overall averages.Not to mention 32,895 price records have the value 0. These outliers could potentially indicate unique or rare instances within the dataset, warranting careful consideration during any analytical or modeling endeavors.

### Hypothesis

Upon analyzing the dataset column by column, several observations and strategies for data preprocessing emerge:

##### Dropping attributes:

1. **ID:** We can safely drop this column as it serves as an index and does not provide meaningful information.
   
2. **VIN:** Since it serves as a unique identifier, we can safely drop this column.

##### Filling impurities:

1. **Year:** We can replace missing values with the most frequent year observed in the dataset.

2. **Manufacturer:** Despite some null values, we can impute them with the most frequently observed manufacturer.

3. **Cylinders:** Convert this column to numerical and replace missing values with the most frequent value, considering the risk of imputing nearly half of null values.

4. **Drive:** While important, this feature has a high percentage of null values. We'll impute them with the most frequent value, ensuring it doesn't exceed 50% of the total dataset.

##### State and Region:

These are categorical variables without null values, so no action is needed.

##### Transmission, Fuel, Title Status, Condition:

For these categorical variables, we will replace missing values with the most frequently occurring value.

###### Type and Paint Color:

We will replace missing values with the most frequently occurring value for each.

##### Size:

With over 50% null values, this column may not provide useful information and could be dropped after further processing.

##### Model:

With a significant number of unique categorical values, we may need to reduce categories or address null values later. 

##### Extensive data preparation:

1. **Price:** As the target variable, some entries have a value of 0, which we will replace with the mean value. Although, certain extreme outliers could impact the mean, which will need to removed before calculating mean.

2. **Odometer:** Missing values can be imputed by averaging the odometer readings of all vehicles. Although, certain extreme outliers could impact the mean, which will need to removed before calculating mean.

Additionally, we'll remove records with incomplete data, retaining only those with meaningful information beyond just the price, state, and region.


## Data Preparation

After our initial exploration and fine tuning of the business understanding, it is time to construct our final dataset prior to modeling. Here, we want to make sure to handle any integrity issues and cleaning, the engineering of new features, any transformations that we believe should happen (scaling, logarithms, normalization, etc.), and general preparation for modeling with sklearn.

![Outliers Output]()


Data preprocessing steps are performed on a dataset containing information about cars. The process involves several key steps to ensure the dataset is cleaned and prepared for further analysis or modeling. Firstly, outliers in the 'odometer' and 'price' columns are identified and removed using both Z-score and interquartile range (IQR) methods. This helps to eliminate extreme values that could skew the analysis or modeling results. Additionally, unnecessary features such as 'VIN' and 'id' are dropped from the dataset to streamline the data.

Following outlier removal and feature dropping, missing data in the dataset are imputed using appropriate strategies. The 'odometer' column's missing values are replaced with the mean, while missing values in the 'year' and 'cylinders' columns are filled with the most frequent values. Furthermore, categorical columns with missing values, including 'transmission', 'fuel', 'title_status', and others, are filled with the mode of each respective column. Finally, the 'model' feature is cleaned by removing models with fewer than 50 occurrences and splitting the remaining models based on space and forward slash separators.

Overall, these preprocessing steps ensure that the dataset is cleaned, standardized, and ready for further analysis or modeling tasks. By addressing outliers, dropping unnecessary features, and imputing missing data, the dataset is better suited for extracting meaningful insights or building predictive models.


Here is the summary info of the clean dataset

![Clean dataset]()

### Preparing dataset for modeling

The dataset is further prepared for modeling by adding a new attribute, 'price_log,' which holds the logarithm of the target variable 'price.' This transformation is applied to mitigate the skewness of the 'price' feature, which can improve the performance of certain machine learning algorithms that assume normality in the target variable distribution.

Following the addition of 'price_log,' categorical features are encoded using target encoding to convert them into numerical representations suitable for machine learning algorithms. The TargetEncoder class from the category_encoders library is utilized for this purpose. Categorical features such as 'manufacturer,' 'model,' 'size,' 'cylinders,' 'fuel,' 'state,' 'region,' 'title_status,' 'transmission,' 'drive,' 'type,' and 'paint_color' are encoded based on their relationship with the target variable 'price.'

Once encoded, the original categorical features are dropped from the dataset, and the encoded versions are concatenated with the existing dataframe. This results in a new dataframe, 'cars_encoded,' with the encoded categorical features and the original numerical features, including 'price_log,' ready for further modeling and analysis.

Additionally, the 'condition' feature, representing the condition of the vehicle, is converted to a numeric scale ranging from 1 (salvage) to 6 (new) to facilitate analysis and modeling. This transformation allows for a more straightforward interpretation of the 'condition' feature's impact on the target variable.

Lastly, a correlation matrix is generated and visualized using a heatmap to examine the relationships between different features in the dataset. This analysis helps identify potential correlations between variables, providing insights into their interdependencies and informing feature selection for predictive modeling.


![corr matrix]()

The correlation matrix, represented visually through a heatmap, provides valuable insights into the relationships between different variables, affirming and shedding light on initial assumptions.

With respect to Price of the cars

The most significant correlations emerge with attributes like year of manufacture, odometer reading, specific models, and vehicle types. Surprisingly, notable correlations also exist with technical aspects such as cylinder count, transmission type, and drive configuration. Conversely, features like condition, fuel type, and title status demonstrate weaker correlations, consistent with the expectation that most vehicles are typically in good condition, fueled by gas, and hold a clean title.

Furthermore, additional noteworthy correlations surface between year and odometer readings, which aligns with the intuitive expectation of odometer values increasing over time. Likewise, associations between model and vehicle type, manufacturer and model, as well as technical attributes and vehicle types, offer deeper insights into the interplay between these characteristics.

In summary, the correlation analysis not only confirms certain initial hypotheses but also unveils nuanced relationships between variables, contributing to a more comprehensive understanding of the factors influencing vehicle prices.

## Modeling

With your (almost?) final dataset in hand, it is now time to build some models. Here, you should build a number of different regression models with the price as the target. In building your models, you should explore different parameters and be sure to cross-validate your findings.

### Linear Regression Model on Log(price)

MSE of Linear Regression model is 0.267159
R2 score of Linear Regression model is 0.582577

Linear regression model to predict the logarithm of the price ('price_log') of cars based on a set of independent variables ('X'). Firstly, the dataset is split into training and testing sets using the train_test_split function from the scikit-learn library. The model is then trained on the training data using the LinearRegression class, which fits a linear model to the data with an intercept. Mean Squared Error (MSE) and R-squared (R^2) are calculated to evaluate the performance of the model on the testing data. MSE quantifies the average squared difference between predicted and actual values, while R^2 indicates the proportion of variance in the dependent variable explained by the model.

Additionally, the code generates predictions for the training data and compares them to the actual prices, creating a DataFrame ('df_predictions') to display the actual and predicted prices. This comparison allows for an assessment of the model's accuracy and its ability to capture the underlying patterns in the data. Furthermore, a scatter plot is created to visualize the relationship between the true values of the logarithm of price and the model's predictions.

Testing regression on the logarithm of price is essential for several reasons. Firstly, it helps address the issue of heteroscedasticity, where the variance of the residuals is not constant across all levels of the independent variables. By taking the logarithm of the price, we stabilize the variance of the target variable, making the model more robust and improving its predictive performance. Additionally, transforming the target variable can lead to a more linear relationship between the predictors and the target, which is a fundamental assumption of linear regression. Overall, testing regression on the log of price allows for a more accurate and interpretable modeling process, enhancing the reliability of the results and facilitating better decision-making in practical applications.

![Price log reg]()

### Linear Regression Model on Real Price values

Building upon the previous linear regression model, but instead of predicting the logarithm of the price ('price_log'), it predicts the actual, real values of the price ('price'). Similar to the previous model, the dataset is split into training and testing sets using the train_test_split function. The model is then trained on the training data using the LinearRegression class, fitting a linear model to the data with an intercept. Mean Squared Error (MSE) and R-squared (R^2) are calculated to evaluate the model's performance on the testing data.

MSE of Linear Regression model is 72129205.489444
R2 score of Linear Regression model is 0.59423

The MSE quantifies the average squared difference between predicted and actual price values, providing insight into the accuracy of the model's predictions. Meanwhile, R^2 indicates the proportion of variance in the dependent variable (price) that is explained by the model. A higher R^2 value suggests that the model can better explain the variability in the data, indicating a better fit.

![real price predictions]()

Furthermore, a scatter plot is generated to visualize the relationship between the true values of the price and the model's predictions. This plot allows for a visual assessment of how well the model's predictions align with the actual prices. By comparing the scatter plots of both the logarithmic and real price models, we can observe any differences in their predictive performance and better understand the effectiveness of each model in capturing the underlying patterns in the data.

### Linear Regression Model with Polynomial Features

Polynomial regression model is implemented to capture potential nonlinear relationships between features and the target variable (price). The process begins by splitting the dataset into training and testing sets using the train_test_split function. The simple_cross_validation function is then defined to iterate through different polynomial degrees (ranging from 1 to 6). For each degree, a pipeline is constructed comprising three steps: polynomial feature transformation, standard scaling, and linear regression. The model is fitted to the training data, and the mean squared error (MSE) is calculated using the testing data.

![best ploy degree]()

During the cross-validation process, the model with the lowest MSE on the testing data is selected as the optimal model. In this case, the model with a polynomial degree of 5 is identified as providing the lowest MSE. Subsequently, a pipeline is created with the optimal polynomial degree, and the model is trained on the training data. The MSE is then computed for both the training and testing data to evaluate the model's performance.

Test MSE: 36017368.65203682
Train MSE: 34966141.06217689

Finally, scatter plots are generated to visualize the relationship between the true values of the price and the model's predictions for both the training and testing datasets. These plots allow for a visual assessment of how well the model's predictions align with the actual prices, providing insight into the model's accuracy and potential overfitting or underfitting issues. Overall, the implementation of polynomial regression enables the model to capture nonlinear patterns in the data, potentially improving its predictive performance compared to a simple linear regression model.

![poly 5 predictions]()

![poly 5 scatter]()


### Sequential Selector with PolynomialFeatures

In this code segment, a feature selection process is performed to identify the most relevant features for predicting the price of cars. Initially, the dataset is split into features (X) and the target variable (y), which is the price of the cars. Then, the dataset is further split into training and testing sets using the train_test_split function with a random state of 42 and a test size of 30%.

Next, polynomial features of degree 5 are generated using the PolynomialFeatures class, capturing potential nonlinear relationships between features. These polynomial features are then transformed into all possible combinations and stored in a DataFrame named all_degree_5_combinations.

![seq select poly 5]()

To perform feature selection, the SequentialFeatureSelector is utilized with a linear regression estimator. This selector evaluates different subsets of features and chooses the subset that minimizes the mean squared error (MSE). The best subset of features is determined based on the negative mean squared error score calculated during cross-validation.

The selected features are then fitted to a Linear Regression model, and the coefficients of the selected features are obtained. These coefficients indicate the strength and direction of the relationship between each feature and the target variable (price). Finally, the top 7 features with the highest coefficients are printed, providing insight into which features are most influential in predicting the price of cars. Overall, this process helps to identify the most important features for building an effective predictive model.

![top 7 seq features]()

### Lasso Regression Model

 Lasso Regression model is implemented and optimized using GridSearchCV. The dataset is prepared by splitting it into features (X) and the target variable (y), which represents the price of cars. Then, the data is further split into training and testing sets using the train_test_split function with a random state of 42 and a test size of 30%.

For the Lasso Regression model, a pipeline is constructed consisting of three steps: polynomial feature generation with a degree of 5, feature scaling using StandardScaler, and the Lasso regression algorithm itself. This pipeline allows for seamless integration of data preprocessing and model training.

GridSearchCV is then employed to search for the optimal value of the alpha hyperparameter, which controls the strength of regularization in the Lasso model. The grid search is performed using a dictionary of alpha values, and the scoring metric chosen for optimization is the negative mean squared error (neg_mean_squared_error) using a 3-fold cross-validation strategy.

{'lasso__alpha': 0.0001}

After fitting the Lasso Regression model to the data, the best hyperparameters obtained from the grid search are printed. Additionally, the mean squared error (MSE) is calculated for both the training and testing sets to evaluate the model's performance. Finally, the actual and predicted values of the target variable from the training set are stored in a DataFrame for further analysis. Overall, this code segment demonstrates the implementation and optimization of the Lasso Regression model for predicting car prices.

Training MSE for lasso model: 66148755.526558325
Testing MSE for lasso model: 66836063.56073399

![Lasso Predictions]()



### Ridge Regression Model with One-Hot Encoding

Ridge Regression model is trained using One-Hot encoding to represent categorical features. The dataset consists of selected columns from the cleaned car data, including features like year, odometer reading, manufacturer, model, cylinders, fuel type, transmission type, drive type, vehicle type, condition, and the target variable price.

First, the data is split into features (X) and the target variable (y). Then, a column transformer is created using the make_column_transformer function from scikit-learn. This transformer applies One-Hot encoding to categorical features like manufacturer, model, cylinders, fuel, transmission, drive, and type, while the condition feature is encoded using an ordinal encoder with predefined categories.

A pipeline is constructed to streamline the preprocessing steps, which include the column transformation, feature scaling using StandardScaler, and the Ridge regression algorithm. The pipeline facilitates the sequential application of these transformations and the training of the model.

![Ridge pipeline]()

The Ridge Regression model is trained on the training data using the fit method of the pipeline. After training, the model's performance is evaluated on both the training and testing sets using mean squared error (MSE) and R-squared (R^2) as evaluation metrics. The MSE indicates the average squared difference between the predicted and actual prices, while R^2 measures the proportion of variance in the target variable that is explained by the model.

MSE for Ridge Regression model : 41004718.461
R^2 for Ridge Regression model : 0.769

Finally, a scatter plot is created to visualize the relationship between the real prices and the model predictions on the testing set. This plot allows for a visual assessment of how well the model's predictions align with the actual prices, providing insights into the model's performance.

![Ridge Predictions]()


## Evaluation

### Permuatation Importance of Ridge Model with one-hot encoding

Permutation importance and coefficient analysis are performed to understand the influential features driving car prices in a Ridge regression model with One-Hot encoding.

First, permutation importance analysis is conducted to determine the importance of each feature in predicting the logarithm of car prices. This analysis ranks the features based on how much the model's performance decreases when the values of each feature are randomly shuffled. The top three influential features identified through permutation importance are model, manufacturer, and transmission, suggesting their significant impact on price prediction.

manufacturer47118178.000 +/- 595651.020
model   38974571.439 +/- 624355.879
transmission268519.246 +/- 16248.478

Next, the coefficients of the Ridge regression model are examined to quantify the effect of each feature on car prices directly. The coefficients represent the change in the logarithm of car prices for a one-unit change in each feature. The top coefficients driving prices higher include the year of the car, specific manufacturers like Chevrolet and Ford, fuel type (diesel), and certain models like Q8 Premium and Super Duty. These coefficients provide insights into the relative importance of different features in determining car prices.

![important Features]()

The analysis suggests that newer model years, specific manufacturers, and certain models tend to command higher prices, while factors like odometer reading and certain manufacturers like Honda and Kia have negative coefficients, indicating lower prices. These findings provide valuable insights for car sellers, buyers, and industry stakeholders, enabling them to make informed decisions regarding pricing, marketing strategies, and product offerings. The combination of permutation importance and coefficient analysis offers a comprehensive understanding of the factors influencing car prices and can guide decision-making processes in the automotive industry.


## Deployment

Improving dataset quality is indeed a critical next step for enhancing model accuracy and reliability. This involves addressing missing values, duplicates, and outliers. Strategies such as imputation, removal, or advanced techniques like predictive modeling to fill in missing values can be employed. Additionally, thorough data cleaning processes can help identify and handle duplicates and outliers appropriately to prevent them from skewing the model's performance.

Further analysis on the impact of mileage on prices could provide valuable insights for sellers, as mileage is a significant factor affecting a car's value. Exploring nonlinear relationships between mileage and prices using techniques like polynomial regression or spline regression can uncover more nuanced patterns and improve the model's predictive power. Additionally, segmenting the dataset based on mileage ranges and analyzing price trends within each segment can provide actionable insights for pricing strategies.

Leveraging the popularity of certain models, as indicated by the "Top 20 Models" chart, can inform pricing strategies by adjusting prices based on demand and market trends. Incorporating external market data, such as sales volumes, competitor pricing, and consumer preferences, can further enhance pricing strategies and help optimize revenue.

Furthermore, developing more nuanced pricing models beyond simple linear or ridge regression can lead to better accuracy and performance. Techniques such as ensemble methods (e.g., random forests, gradient boosting), neural networks, or Bayesian regression can capture complex relationships between features and prices more effectively. Model evaluation and validation using techniques like cross-validation and robust performance metrics are also crucial for ensuring model reliability and generalizability.

Conducting comprehensive market analyses, including competitor analysis, customer segmentation, and demand forecasting, can provide valuable insights into market dynamics and help identify untapped opportunities for pricing optimization. Continuous monitoring of market trends and consumer behavior allows for timely adjustments to pricing strategies, ensuring competitiveness and profitability in the dynamic automotive market landscape.

In summary, focusing on dataset quality improvement, conducting further analysis on mileage impact, leveraging model popularity insights, exploring advanced modeling techniques, and conducting comprehensive market analyses are recommended next steps to enhance pricing strategies and decision-making processes in the automotive industry.

