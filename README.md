# Student Performance ML

## Project Overview

This project uses **Machine Learning** to predict whether a student will **Pass or Fail** based on academic and behavioral information.

A **Decision Tree Classifier** is used to train the model and predict the final result.

## Dataset

The dataset used in this project is:

`student_performance_ml.csv`

The dataset contains the following features:

| Feature              | Description                                        |
| -------------------- | -------------------------------------------------- |
| StudyHours           | Number of hours a student studies per day          |
| Attendance           | Percentage of class attendance                     |
| PreviousScore        | Marks obtained in the previous examination         |
| AssignmentsCompleted | Number of assignments completed                    |
| SleepHours           | Average number of hours the student sleeps per day |
| FinalResult          | Target variable: 1 = Pass, 0 = Fail                |

## Machine Learning Algorithm

**Decision Tree Classifier**

The Decision Tree learns patterns from the student data and predicts whether a student will pass or fail.

## Project Steps

1. Load the dataset using Pandas
2. Perform basic data analysis
3. Visualize StudyHours using a histogram
4. Split the dataset into training and testing data
5. Train the Decision Tree model
6. Predict results for test data
7. Calculate model accuracy
8. Generate a confusion matrix
9. Display the final model conclusion

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

## Python Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
```

## Model Evaluation

The model is evaluated using:

### Accuracy

Accuracy measures the percentage of correctly predicted results.

### Confusion Matrix

The confusion matrix shows:

* True Positive
* True Negative
* False Positive
* False Negative

It helps understand the detailed performance of the classification model.

## Project Structure

```text
StudentPerformance/
│
├── StudentPerformance.py
├── student_performance_ml.csv
└── README.md
```

## How to Run

### 1. Install required libraries

```bash
pip install pandas matplotlib scikit-learn
```

### 2. Keep the files in the same folder

```text
StudentPerformance.py
student_performance_ml.csv
```

### 3. Run the program

```bash
python StudentPerformance.py
```

## Output

The program displays:

* Dataset information
* Data analysis results
* StudyHours histogram
* Predicted values
* Actual values
* Testing accuracy
* Confusion matrix
* Final model conclusion

## Conclusion

The project demonstrates how a **Decision Tree Classifier** can be used to analyze student performance data and predict whether a student will **Pass or Fail**.

The model uses factors such as **StudyHours, Attendance, PreviousScore, AssignmentsCompleted, and SleepHours** to make predictions.
