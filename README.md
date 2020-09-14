# Flipr-Hackathon-6.0-ML
This is the solution presented by Team Chunchunmaaru on the two tasks assigned during the hackthon.

The  tasks are as follow :

Part -01:
The objective of the first part of the problem statement is to predict the Covid Cases of a
City on 1st September 2020. The output file 01 should contain only City and the respective
Covid Cases for the test data.

Part -02:
The Foreign Visitors of a city is a time-dependent parameter, for which you have to come up
with a Time-series prediction model. Using the Foreign Visitors predicted by the model, you
need to calculate the Covid Cases on 1st Oct 2020 for every City in the test data. . The
output file 02 should contain only City and the respective Covid Cases on 1st October.

APPROACH TASK-1 :

For the first task, we started out with some feature engineering and tried many standard Machine learning algorithms and decided it would be best  to use Fully Connected Neural Network / Deep Neural Net for this regression Problem.

Feature Engineering included dropping whole columns named:  City, Type, Population (2011) and State columns, as they were irregular in both, test set and train set. Initially, we considered replacing State and Type with categorical values, but that didn't work because of the presence of uncommon values in them.

Next, we converted comma separated numbers to float and while we were at it we also took care of nan values by replacing them with the median of the column they were in.

So, basically we chose the FNN/DNN for this regression task, as they are very powerful tools in the field of machine learning. We selected this model based on hit and trial for this particular task. We, also out of curiosity, looked at another interesting parameter - how many predictions only differ by 1000 mark .

Our approach is suitable because, the model was overfitting the train data, for other ML algo's but with the help of early stopping we managed to reduce this overfitting to a great extent.

We made a fairly simply Neural Net and trained it with Adam Optimizer and tried a bunch of different structures of neural nets and hyperparameters. The best response was obtained as the one presented in this code. Then we predicted the COVID-19 cases as per the problem statement.


APPROACH TASK 2 :

For the second task we were asked to use foreign visitor as a time dependent parameter and first calculate the number of foreign visitors on 31st Sept and COVID-19 caseload till 1st Oct.

For the first part of the problem, we used an LSTM. A fairly simple one with adam optimizer, we extracted the no. of foreign visitors in each month for each city described, and used this (4,1) shaped array as a input to predict total foreign visitor till 31st Sept / before 1st Oct. 

There was a catch for the nan values we didn't use masking in the model due to time constraints, though I prefer masking and we just simply replaced the nan values with 0. 

Then on analysing the trend we found that max (sept_val+500,predicted_val ) should be taken, as 500 is the cap for minimum change in from one month to another. This was done so as to counter the 0 we had used earlier due to time constraints.

Then we replaced the foreign visitor column in the test set and then performed pre-processing on it to obtain the predicted values like before. Again LSTM / RNN are one of the best tools when we have dependency on previous states and we tried to exploit it to the fullest extent.
