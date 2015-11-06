# CarAuctionEvaluator
The goal of this project is to use machine learning to build a predictive model to help auto dealers avoid purchasing potentially bad vehicles. The model will be based on a set of 72,000 historical records that contains 32 attributes as well as a unique id for each purchase and a label indicating if the vehicle was a bad buy or not.

This project is for the Kaggle [*Don't Get Kicked!*](https://www.kaggle.com/c/DontGetKicked) challenge.

## Preliminary Analysis
- Bad Buy Percentage: 12.30%
- Years range from 2001 to 2010. The median year was 2005 and the mean year was 2005. The most popular year is 2006.
- Age ranges from 0 to 9 years old. The median age was 4 and the mean age was 4.
- The most popular make is CHEVROLET.
- The most popular color is SILVER.
- Odometer readings range from 4825 to 115717 miles. The median reading was 73363 and the mean reading was 71502.

## Strategies
- Random Forest
- Naive Bayes
- Neural Network
- Decision Tree
- Adaboost (base=Decision Tree)

I ultimately chose to use the Adaboost algorithm as decision trees were producing the best results. I focused on maximizing the accuracy of predicting bad buys without taking a huge hit to the accuracy of the good buys.


## Dependencies
- [Pandas](http://pandas.pydata.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [scikit-neuralnetwork](https://github.com/aigamedev/scikit-neuralnetwork)
- [matplotlib](http://matplotlib.org)
- [prettytable](https://code.google.com/p/prettytable/)
- [numpy](http://numpy.org)