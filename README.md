# ML_Model_for_Halloween_Candy_Power_Ranking_Classification
Author: Parameswara rao

Date: 04 April 2020

I build a simple model using candy-data.csv data set for predicting candy is chocolate or not based on its other features. This project is Organised as follows:
(a) extract featues from data set.
(b) data preprocessing and split into train and test data for a given model.
(C) hyper parameters tuning
(d) basic logistic regression classification model
(e) summary of model output.

candy-data.csv: ncludes attributes for each candy along with its ranking. For binary variables, 1 means yes, 0 means no. The  data contains the following fields:
1. chocolate: Does it contain chocolate?
2. fruity: Is it fruit flavored?
3. caramel: Is there caramel in the candy?
4. peanutalmondy: Does it contain peanuts, peanut butter or almonds?
5. nougat: Does it contain nougat?
6. crispedricewafer: Does it contain crisped rice, wafers, or a cookie component?
7. hard: Is it a hard candy?
8. bar: Is it a candy bar?
9. pluribus: Is it one of many candies in a bag or box?
10. sugarpercent: The percentile of sugar it falls under within the data set.
11. pricepercent: The unit price percentile compared to the rest of the set.
12. winpercent: The overall win percentage according to 269,000 matchups.


This dataset is Copyright (c) 2014 ESPN Internet Ventures and distributed under an MIT license. Check out the analysis and write-up here: The Ultimate Halloween Candy Power Ranking. Thanks to Walt Hickey for making the data available.


Result: 
Accuracy on test data for a given model is 0.9615384615384616
