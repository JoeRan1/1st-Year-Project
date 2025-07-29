# 1st-Year-Project (Real Estate Agency)
Dataset: https://drive.google.com/file/d/11-ZNNIdcQ7TbT8Y0nsQ3Q0eiYQP__NIW/view
Serialized model (pickle format): https://drive.google.com/file/d/1f2m1Z_YtEybiU0kgoCH5OOaKSGTQYGu9/view?usp=sharing

-
The task implied to create a prediction model for House Prices. Dataset turned out to be very messy, initially including 18-20 features with more than 370K rows.
I started EDA with basic .head(), .info() and .describe() functions. Visualizations did not give expected effects, dataset is full of not normally distributed columns with very big range. So it required to get rid of outliers, wrong values, and other elements.

During the Feature Engineering proccess I disclosed that a column 'Homefacts' includes a dictioanry formatted data with several parameters

