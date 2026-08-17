# #This is the single structured program that performs:

# Dataset loading

# Data analysis

# Visualization

# Train-test split

# Model training

# Prediction

# Accuracy calculation

# Confusion matrix

# Final conclusion

import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


#step 1: Load the data 
#----------------------------------------------------------
# Function Name : LoadData
# Description   : Load student performance data from CSV file
# Input         : None
# Output        : DataFrame
# Author        : Prapti Kulkarni
# Date          : 17/08/2026
#---------------------------------------------------------

def LoadData():
    df = pd.read_csv("student_performance_ml.csv")
    print("Dataset loaded successfully")
    print(df.head())
    return df

#----------------------------------------------------------
# Function Name : DataAnalysis
# Description   : Display basic information and statistics of dataset
# Input         : DataFrame
# Output        : None
# Author        : Prapti Kulkarni
# Date          : 17/08/2026
#----------------------------------------------------------
def DataAnalysis(df):
    print("Total rows: ",df.shape[0])
    print("Total column: ",df.shape[1])

    print("Avarage StudyHours: ",df["StudyHours"].mean())
    print("Avarage Attendance: ,",df["Attendance"].mean())

#----------------------------------------------------------
# Function Name : Visualization
# Description   : Display histogram of StudyHours
# Input         : DataFrame
# Output        : Histogram
# Author        : Prapti Kulkarni
# Date          : 17/08/2026
#----------------------------------------------------------

def Visualization(df):
    plt.hist(df["StudyHours"],bins =5,edgecolor="black",rwidth=0.9)
    plt.title("Histogram of StudyHours")
    plt.xlabel("StudyHours")
    plt.ylabel("Number of Students")
    plt.show()

#----------------------------------------------------------
# Function Name : splitdata
# Description   : Perform train-test splitting
# Input         : DataFrame
# Output        : Training and testing data
# Author        : Prapti Kulkarni
# Date          : 17/08/2026
#----------------------------------------------------------

def Splitdata(df):
    X = df[["StudyHours", "Attendance", "PreviousScore",
            "AssignmentsCompleted", "SleepHours"]]
    Y =df["FinalResult"]

    X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=42)

    print("Dataset splitting completed successfully")

    return X_train,X_test,Y_train,Y_test

#----------------------------------------------------------
# Function Name : TrainModel
# Description   : Perform Decision Tree model training
# Input         : Training features and labels
# Output        : Trained model
# Author        : Prapti Kulkarni
# Date          : 17/08/2026
#----------------------------------------------------------

def TrainModel(X_train,Y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train,Y_train)

    print("Model Trained succressfully")

    return model

#----------------------------------------------------------
# Function Name : evaluatemodel
# Description   : Predict and evaluate the model
# Input         : Model, testing data and labels
# Output        : Accuracy and predictions
# Author        : Prapti Kulkarni
# Date          : 17/08/2026
#----------------------------------------------------------

def Evalautemodel(model,X_test,Y_test):
    y_pred = model.predict(X_test)

    print("Actual values:")
    print(Y_test.values)

    print("Predicted values :")
    print(y_pred)

    accuracy = accuracy_score(Y_test,y_pred)

    print("Accuracy is: ",accuracy*100,"%")

    cm = confusion_matrix(Y_test,y_pred)
    print("Confusion matrix : ")
    print(cm)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    display.plot()
    plt.title("confusion matrix")
    plt.show()
    return accuracy

#----------------------------------------------------------
# Function Name : main
# Description   : Entry point function
# Input         : None
# Output        : None
# Author        : Prapti Kulkarni
# Date          : 17/08/2026
#----------------------------------------------------------

def main():
    print("Step 1 : Dataset Loading")
    df =LoadData()

    print("\nStep 2 : Data Analysis")
    DataAnalysis(df)

    print("\nStep 3 : Visualization")
    Visualization(df)

    print("\nStep 4 : Train-Test Split")
    X_train,X_test,Y_train,Y_test=Splitdata(df)

    print("\nStep 5 : Model Training")
    model =  TrainModel(X_train,Y_train)

    accuracy = Evalautemodel(model,X_test,Y_test)
    print("\nStep 9 : Final Conclusion")

    if accuracy >= 0.75:
        print("Model performance is good")
    else:
        print("Model performance needs improvement")

if __name__=="__main__":
    main()