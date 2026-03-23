Pseudo code:

Read Train Data
Remove unwated columns

From the selected columns
Seperate Categorical and continuous columns

<< Work on Categorical columns >>
Do 1-Hot Encoding use GetDummies()
Transofrm categorical columns into 1Hot encoded columns

#EDA
#Imputation (Filling Missing Data)
Age.inull() = Age.Mean() #Average

dt.DecissionTreeClassifier(with Max. no. of columns)
dt.fit()



Print decision trees
Expplain the decision Tree insights with small column data sets

Read Test data with same columns of Train data
y_survived=dt.predict()

Prepare a new Excel with TestPassenderID and Survived o/p
pd.writetoCSV(TestID, y_Survived)

Upload to the Kaggle
 and check your prediction
