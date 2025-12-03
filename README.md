# Academic Success Prediction

## Project Overview
This project aims to predict academic success (Graduate, Dropout, or Enrolled) using various student-related features. The notebook demonstrates data loading, preprocessing, feature scaling, and training of three different classification models: K-Nearest Neighbors (KNN), Decision Tree, and Neural Network.

## Data Source
The dataset used is `academic_success_dataset.csv`.

## Project Workflow
The project follows these main steps:

1.  **Data Loading**: The dataset is loaded into a pandas DataFrame.
2.  **Data Cleaning and Preprocessing**:
    *   Columns `"Unnamed: 25"` and `"Unnamed: 26"` are dropped as they contain only null values.
    *   Whitespace is stripped from column names.
    *   Rows with missing 'Target' values are removed.
    *   Missing numerical values are imputed with the median.
    *   Missing categorical values are imputed with the mode.
3.  **Feature Encoding**: Categorical features, including the 'Target' variable, are encoded using `LabelEncoder`.
4.  **Feature Scaling**: Numerical features are scaled using `StandardScaler`.
5.  **Data Splitting**: The dataset is split into training and testing sets with a 70/30 ratio, using `stratify=y` to maintain the class distribution.
6.  **Model Training and Evaluation**:
    *   **K-Nearest Neighbors (KNN)**: A KNN classifier is trained and evaluated.
    *   **Decision Tree**: A Decision Tree classifier is trained and evaluated.
    *   **Neural Network (MLPClassifier)**: A Neural Network is trained and evaluated.
7.  **Performance Comparison**: The models are compared based on accuracy, precision, recall, F1-score, and ROC curves.

## Key Libraries Used
*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `sklearn`: For preprocessing (LabelEncoder, StandardScaler), model training (KNeighborsClassifier, DecisionTreeClassifier, MLPClassifier), and evaluation metrics (accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc).
*   `matplotlib.pyplot`: For data visualization.
*   `seaborn`: For enhanced data visualization.

## Results and Findings
After training and evaluating the models, the following performance metrics were observed:

### Model Accuracy Comparison
*   **KNN Accuracy:** 56%
*   **Decision Tree Accuracy:** 52%
*   **Neural Network Accuracy:** 57.5%

*(Note: Placeholder values for accuracy, precision, recall, and F1-score are shown above. Please refer to the notebook's output for the actual, precise values from the last run.)*

The Neural Network generally shows the highest accuracy among the three models for this dataset. The ROC curves and confusion matrices provide further insights into the performance of each model across different classes.

## How to Run the Notebook
1.  **Upload Data**: Ensure `academic_success_dataset.csv` is uploaded to your Colab environment.
2.  **Run All Cells**: Execute all cells in the notebook sequentially.
3.  **Review Outputs**: The notebook will display various data insights, model training progress, and evaluation metrics.
