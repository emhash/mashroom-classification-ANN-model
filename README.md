# Mushroom Classification: Binary Classification Analysis

## Project Overview
This project implements a binary classification model to distinguish between edible and poisonous mushrooms using a dataset of 8,124 samples with 23 categorical features. The analysis includes data preprocessing, exploratory data analysis, neural network implementation, hyperparameter tuning, and comparison with traditional machine learning algorithms.

## Dataset
- **Source**: Mushrooms dataset loaded from Google Drive
- **Size**: 8,124 samples × 23 features
- **Target Variable**: `class` (p = poisonous, e = edible)
- **Features**: All categorical variables including cap-shape, cap-surface, cap-color, bruises, odor, gill-attachment, etc.

## Methodology & Workflow

### 1. Data Loading & Initial Exploration
- Loaded mushroom dataset from Google Drive using pandas
- Examined dataset structure and dimensions (8,124, 23)
- Created pairplot visualization to understand feature distributions by class
- Verified data balance between poisonous (p) and edible (e) mushrooms

### 2. Data Preprocessing
- **One-Hot Encoding**: Applied `pd.get_dummies()` to convert all categorical variables to numerical format
- **Target Encoding**: Converted target variable ('p' → 0, 'e' → 1) for binary classification
- **Feature Engineering**: Dropped redundant class columns (class_e, class_p) after encoding
- **Data Splitting**: Used `train_test_split` with 80/20 ratio (random_state=0)
- **Feature Scaling**: Applied MinMaxScaler before neural network training

### 3. Neural Network Implementation

#### Initial Model Architecture:
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

- **Optimizer**: SGD
- **Loss Function**: binary_crossentropy
- **Training**: 15 epochs with validation split
- **Results**: Achieved 99.14% test accuracy

### 4. Model Optimization
- **Hyperparameter Tuning**: Implemented GridSearchCV with KerasClassifier wrapper
- **Optimized Architecture**: 
  - Input layer: 64 neurons (ReLU)
  - Hidden layer: 32 neurons (ReLU) 
  - Dropout layers: 0.4 and 0.3 rates
  - Output: Single neuron (sigmoid)
- **Optimizer Comparison**: Tested different optimizers through grid search

### 5. Model Evaluation & Visualization
- **Confusion Matrix**: Generated and visualized classification results
- **ROC Curve**: Plotted ROC curve and calculated AUC score
- **F1-Score**: Computed F1-score for model performance assessment
- **Training History**: Plotted accuracy and loss curves for training/validation

### 6. Comparative Analysis
Implemented and compared multiple algorithms:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**

Each model evaluated using:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix
- ROC curve analysis

### 7. Advanced Model Analysis
- **Random Forest Deep Dive**: 
  - Calculated log loss for train/test sets
  - Plotted accuracy comparison between train/test
  - Performed cross-validation with KFold (5 splits)
  - Generated learning curves to analyze model performance vs. training size

## Key Results
- **Neural Network**: 99.14% accuracy
- **All Models**: Achieved perfect or near-perfect classification (100% accuracy)
- **Cross-validation**: Consistent high performance across all folds
- **No Overfitting**: Learning curves show good generalization

## Technical Implementation
- **Libraries**: pandas, numpy, scikit-learn, tensorflow/keras, matplotlib, seaborn, xgboost
- **Encoding**: pandas get_dummies for categorical variables
- **Scaling**: MinMaxScaler for neural network preprocessing
- **Validation**: Train/test split with cross-validation for robust evaluation
- **Visualization**: Comprehensive plotting for model interpretation

## Files Structure
```
├── mashroom-classification-deep-learning.ipynb    # Complete analysis notebook
├── README.md                                      # Project documentation
```

## How to Run
1. Open the Jupyter notebook
2. Run cells sequentially from data loading through model evaluation
3. All dependencies and data loading are handled within the notebook
4. Results and visualizations will be generated inline

This project demonstrates end-to-end machine learning workflow from data preprocessing through model deployment and evaluation, showcasing both deep learning and traditional ML approaches for binary classification problems.
