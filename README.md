# Titanic-Survival
Machine Learning in Python, Jupyter, Keras, Tensorflow, SVM, Decision Tree, Random Forest Regression

--

This is my first dip in the `kaggle` pool. I try to answer the Titanic problem primarily to get a feel of the competitions, to see if what I've learned in  the Self-Driving Car Nanodegree is sufficient for ML jobs, to learn more, practice more, and most importantly, have fun.

The Titanic algorithm will give a prediction of who among a dataset of passengers will have survived the disaster. On my first try, I used CNN which I thought would give the most accurate result. I was so wrong and got a very bad result.

07/31/2017
Accuracy = 0.3792
It runs. But it's not learning.

I tried tweaking the model and was able to improve it a bit.

08/01/2017
Accuracy = 0.6777
A little better.

I tried to follow a tutorial on `Youtube` about ML using the Titanic dataset as example. There were plenty but I followed the one by **Ju Liu** from NoSlidesConference. Love his accent! He used a `Linear Regression` classifier with plenty of visualizations. I learned `Panda` and how easier it is than `matplotlib`. I realized the importance of data exploration and I tried my hand on using other classifiers like the `SVM` and `Decision Tree`, both of which gave a higher score than Linear Regression.

08/03/2017
Accuracy = 0.79
Improving.

I found another `Youtube` video by **Mike Bernico** on the Titanic problem using `Random Forest Regression` in a clear step by step on the whys and hows of optimizing each process. It was awesome! He wrote his code using a Jupyter notebook so I just followed suite. I had to tweak the code a bit to make it work since (I think) he was using Python2. 

08/04/2017
Accuracy = 0.874
Maybe the highest.

I'm trying to find what else I can tweak to raise the accuracy. I tried adding a `cross-validation` but I guess you don't need that with Random Forest. Then I tried `Standard Scaling` which improved the results a little bit (around 5%). I placed a `Binarizer` on my predictions to narrow the results to ones and zeros when I submit it. Well, here goes nothing!

08/05/2017
Public Leaderboard Accuracy = 0.76555
Needs improvement

I can still play with the name titles and cabin numbers as well as combinations of `Parch` and `SibSp`, and combine them with `Fare`. But I'm excited to start with Computer Vision so maybe till next time when I have free time.
