<<<<<<< HEAD
=======

>>>>>>> a4b0100 (Initial commit with all project files)
# Heart Disease Risk Prediction Using KNN Algorithm

## Introduction
This project aims to predict heart disease risk using the K-Nearest Neighbors (KNN) algorithm. Heart disease is a leading cause of death globally, and effective prediction methods are crucial for early intervention. The KNN algorithm, known for its simplicity and effectiveness in classification problems, is employed to develop a predictive model using a publicly available dataset.

## State of the Art
Several machine learning models are commonly used for predicting heart disease, including:
- Logistic Regression
- Support Vector Machines
- Decision Trees
- Neural Networks

Each model has its strengths and weaknesses. KNN is chosen for its ease of implementation, effectiveness with smaller, cleaner datasets, and interpretability.

## Methodology
The project uses the Python programming language, leveraging libraries such as:
- **Pandas** for data manipulation
- **Scikit-learn** for machine learning algorithms
- **Matplotlib** and **Seaborn** for visualization

### Data Preprocessing
1. **Encoding Categorical Variables**: Features like sex, chest pain type, and thalassemia are converted to numerical values using `LabelEncoder`.
2. **Data Splitting**: The dataset is divided into training (80%) and testing (20%) sets to validate the model's performance.

### Model Implementation and Tuning
- **Initial Model**: Implemented with k=3.
- **Hyperparameter Tuning**: GridSearchCV is used to identify the optimal number of neighbors (k), weighting method, and distance metric.

### Tools Used
- **Python**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**

## Results
- **Initial Model**:
 - Accuracy: 90.24%
 - Precision: 89.52%
 - Recall: 91.26%
 - F1 Score: 90.38%
- **Optimized Model**: k=9, 'manhattan' distance metric, 'distance' weights.
 - Achieved perfect scores (1.0) for accuracy, precision, recall, and F1 score on the test data.
 - Cross-validation mean accuracy: 99.71%, indicating consistency.

### Visualizations
- **Confusion Matrix**: Displays true positives, true negatives, false positives, and false negatives for both initial and optimized models.
- **Model Accuracy vs. Number of Neighbors**: Indicates the highest cross-validated accuracy with 9 neighbors.
- **Cross-Validation Accuracy Scores**: Reflects high and stable accuracy across all folds, suggesting good generalization.

## Discussion
The KNN model demonstrated high performance in predicting heart disease risk. Hyperparameter tuning significantly improved the model's accuracy. The results indicate the model's potential in clinical decision support and healthcare resource allocation.

## Implications
- **Clinical Decision Support**: Helps healthcare professionals identify high-risk patients for early intervention.
- **Healthcare Resource Allocation**: Allows more effective allocation of resources, focusing on preventative measures for high-risk patients.
- **Further Research**: Provides a baseline for future research and potential development of more sophisticated models.

## Conclusion
The KNN algorithm has proven to be an effective tool for predicting heart disease risk. The project underscores the importance of hyperparameter tuning and data preprocessing. Future work could explore ensemble methods and more complex algorithms like deep learning to enhance prediction accuracy.

## Contact
For any inquiries, please contact:

Abdul Rehman Rattu

Email: [rattu786.ar@gmail.com](mailto:rattu786.ar@gmail.com)

LinkedIn: [Abdul Rehman Rattu](https://www.linkedin.com/in/abdul-rehman-rattu-395bba237)
