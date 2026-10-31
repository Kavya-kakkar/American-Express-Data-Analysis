# American Express Data Analysis & Deep Learning

An end-to-end machine learning project focused on preparing structured financial data and developing an Artificial Neural Network (ANN) for binary classification using Python and TensorFlow/Keras.

## Project Overview

This project explores a structured machine learning workflow using tabular data from the American Express dataset. It focuses on transforming raw data into a format suitable for deep learning, handling categorical and numerical variables, and building an Artificial Neural Network to predict target outcomes.

The primary objective is to understand and implement the complete data preprocessing and model development lifecycle, from raw dataset ingestion to model training, prediction, and performance evaluation.

The project demonstrates how data transformation, feature scaling, and neural network architectures can be combined to solve binary classification problems involving structured data.

## Key Highlights

- End-to-end data preprocessing and deep learning workflow.
- Categorical feature transformation using Label Encoding and One-Hot Encoding.
- Numerical feature standardization using StandardScaler.
- Reproducible training and testing split.
- Artificial Neural Network implementation using TensorFlow/Keras.
- Binary classification using sigmoid activation.
- Model training across 120 epochs.
- Performance evaluation using a confusion matrix and accuracy score.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Data Preprocessing | LabelEncoder, ColumnTransformer, OneHotEncoder |
| Feature Scaling | StandardScaler |
| Model Evaluation | Scikit-learn Metrics |

## Dataset Overview

The project uses a structured CSV dataset named `american.csv`, containing demographic, geographic, and behavioral attributes.

The dataset is separated into:

- **Independent Variables (X):** Input features used by the neural network.
- **Dependent Variable (y):** Target column used for binary classification.

The preprocessing workflow converts categorical and numerical data into a consistent numerical representation suitable for neural network training.

**Note:** The specific target variable and dataset description should be verified against the original dataset before interpreting the model as a customer churn, credit risk, or fraud prediction system.

## Machine Learning Workflow

### 1. Environment Setup and Data Ingestion

Established the development environment by importing the required Python libraries.

- Imported NumPy and Pandas for numerical operations and data manipulation.
- Imported TensorFlow/Keras for deep learning implementation.
- Loaded the dataset from `american.csv`.
- Separated input features from the target variable.

### 2. Data Preprocessing and Encoding

Transformed raw categorical and numerical attributes into a machine-learning-compatible format.

**Categorical Feature Encoding**

- Label Encoding: Converted binary categorical variables into numerical representations.
- One-Hot Encoding: Transformed multi-category geographic variables into separate numerical columns.
- ColumnTransformer: Applied transformations to the appropriate feature columns.

**Train-Test Split**

Divided the dataset into training and testing subsets using an 80:20 ratio.

A fixed random seed was used to support reproducibility.

### 3. Feature Scaling

Applied StandardScaler to numerical input features to standardize their distributions.

The transformation follows:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) represents the original feature value.
- \(\mu\) represents the training-set mean.
- \(\sigma\) represents the training-set standard deviation.
- \(z\) represents the standardized feature value.

The scaler was fitted using training data and applied to the testing data to avoid data leakage.

### 4. Artificial Neural Network Architecture

Developed a sequential Artificial Neural Network using TensorFlow/Keras.

The architecture consists of:

| Layer | Configuration | Purpose |
|---|---|---|
| Input | Preprocessed feature dimensions | Receives input data |
| Hidden Layer 1 | ReLU activation | Learns nonlinear patterns |
| Hidden Layer 2 | ReLU activation | Captures additional feature relationships |
| Output Layer | 1 neuron, Sigmoid activation | Produces binary classification probability |

**Activation Functions**

- ReLU: Introduces non-linearity into hidden layers.
- Sigmoid: Converts the output into a probability between 0 and 1.

The output probability is used to classify observations into the two target classes.

### 5. Model Compilation and Training

Configured and trained the ANN using the following parameters:

| Parameter | Configuration |
|---|---|
| Framework | TensorFlow/Keras |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Evaluation Metric | Accuracy |
| Training Epochs | 120 |
| Batch Size | 32 |
| Output Activation | Sigmoid |

The model was trained using the training dataset, with loss and accuracy monitored throughout the training process.

### 6. Prediction and Model Evaluation

Used the trained neural network to generate predictions on unseen test data.

The predicted probabilities were converted into binary class labels using a threshold of 0.5.

Model performance was evaluated using:

- **Accuracy Score:** Measures the overall proportion of correctly classified observations.
- **Confusion Matrix:** Provides a breakdown of true positives, true negatives, false positives, and false negatives.

These metrics provide an initial assessment of the model's classification performance.

## Key Technical Learnings

This project provided practical experience with several important data science and deep learning concepts:

1. Transforming raw tabular datasets into machine-learning-ready features.
2. Understanding the differences between Label Encoding and One-Hot Encoding.
3. Applying feature scaling to improve neural network training.
4. Understanding the role of activation functions in ANN architectures.
5. Implementing binary classification using TensorFlow/Keras.
6. Understanding the relationship between loss functions, optimizers, and model training.
7. Evaluating classification predictions using confusion matrices and accuracy.

## Repository Structure

```text
American-Express-Data-Analysis/
│
├── Data_Preprocessing_Template.ipynb
│   # Data preprocessing and ANN implementation
│
├── american.csv
│   # Input dataset (if locally available)
│
└── README.md
    # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook
- Required Python libraries

### 1. Clone the Repository

```bash
git clone https://github.com/Kavya-kakkar/American-Express-Data-Analysis.git

cd American-Express-Data-Analysis
```

### 2. Install Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow jupyter matplotlib seaborn
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Data_Preprocessing_Template.ipynb` and execute the notebook cells sequentially.

**Dataset Requirement:** Ensure that `american.csv` is available at the file path referenced in the notebook before execution.

## Future Improvements

Potential improvements to extend the current implementation include:

- Experimenting with different ANN architectures and hidden-layer configurations.
- Applying dropout and regularization to reduce overfitting.
- Implementing early stopping and learning-rate scheduling.
- Comparing ANN performance against Logistic Regression, Random Forest, and XGBoost.
- Evaluating additional metrics such as Precision, Recall, F1-score, and ROC-AUC.
- Performing hyperparameter optimization.
- Building an interactive prediction interface.
- Deploying the trained model through a FastAPI REST API.

## Conclusion

This project demonstrates a structured approach to preparing tabular data and implementing deep learning for binary classification.

By combining categorical encoding, numerical feature scaling, and an Artificial Neural Network, the workflow covers essential stages of a practical machine learning pipeline.

It also provides hands-on experience with TensorFlow/Keras and demonstrates how preprocessing decisions influence the quality and reliability of machine learning models.

---

**Author:** [Kavya Kakkar](https://github.com/Kavya-kakkar)

**Repository:** [American Express Data Analysis](https://github.com/Kavya-kakkar/American-Express-Data-Analysis)

**Technologies:** Python | Pandas | NumPy | Scikit-learn | TensorFlow | Keras | Deep Learning
