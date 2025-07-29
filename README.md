# 1st-Year-Project (Real Estate Agency)
Dataset: https://drive.google.com/file/d/11-ZNNIdcQ7TbT8Y0nsQ3Q0eiYQP__NIW/view

Serialized model (pickle format): https://drive.google.com/file/d/1f2m1Z_YtEybiU0kgoCH5OOaKSGTQYGu9/view?usp=sharing


The task involved building a predictive model for house prices. The dataset was quite messy, initially containing 18–20 features and over 370,000 rows.

I began exploratory data analysis (EDA) using basic functions such as .head(), .info(), and .describe(). However, the visualizations didn't yield the expected insights because many columns were not normally distributed and had extremely wide value ranges. As a result, it was necessary to remove outliers, incorrect values, and other anomalies.

During the feature engineering process, I discovered that the 'Homefacts' column contained dictionary-formatted data with several parameters, including price per square foot. Since this introduced a risk of data leakage—especially given that the dataset already included a 'sqft' column—I removed that information to better reflect a real-world scenario. I retained only the elements that were useful and did not pose a leakage risk.

I created a separate folder (titled "Visualizations") to store helpful plots used for identifying outliers and informing decisions on normalization and scaling. I also noticed that some houses were labeled with a status of 'for rent', which could negatively affect the model's performance—given that the goal was to predict house prices, not rental values—so I excluded those entries.

Following feature engineering, I applied statistical tests to investigate whether the target variable (house price) significantly depended on location (e.g., states). Both Chi-Square and ANOVA tests confirmed a meaningful relationship.

I then split the dataset into features (X) and target (y), implemented preprocessing with a custom transformer, and embedded the entire process into model pipelines for efficient training and evaluation. I experimented with several predictive models, and ultimately, the Random Forest model delivered the best performance.

 
