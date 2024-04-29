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

#### Dropping attributes:

1. **ID:** We can safely drop this column as it serves as an index and does not provide meaningful information.
   
2. **VIN:** Since it serves as a unique identifier, we can safely drop this column.

#### Filling impurities:

1. **Year:** We can replace missing values with the most frequent year observed in the dataset.

2. **Manufacturer:** Despite some null values, we can impute them with the most frequently observed manufacturer.

3. **Cylinders:** Convert this column to numerical and replace missing values with the most frequent value, considering the risk of imputing nearly half of null values.

4. **Drive:** While important, this feature has a high percentage of null values. We'll impute them with the most frequent value, ensuring it doesn't exceed 50% of the total dataset.

#### State and Region:

These are categorical variables without null values, so no action is needed.

#### Transmission, Fuel, Title Status, Condition:

For these categorical variables, we will replace missing values with the most frequently occurring value.

#### Type and Paint Color:

We will replace missing values with the most frequently occurring value for each.

#### Size:

With over 50% null values, this column may not provide useful information and could be dropped after further processing.

#### Model:

With a significant number of unique categorical values, we may need to reduce categories or address null values later. 

#### Extensive data preparation:

1. **Price:** As the target variable, some entries have a value of 0, which we will replace with the mean value. Although, certain extreme outliers could impact the mean, which will need to removed before calculating mean.

2. **Odometer:** Missing values can be imputed by averaging the odometer readings of all vehicles. Although, certain extreme outliers could impact the mean, which will need to removed before calculating mean.

Additionally, we'll remove records with incomplete data, retaining only those with meaningful information beyond just the price, state, and region.

