# 🧠 Stroke Prediction: Gaussian Naive Bayes Classifier

## Project Overview

This project implements a machine learning solution to predict the risk of a patient experiencing a **stroke** based on key health and lifestyle indicators. This is a crucial **binary classification** task in the medical field, where the cost of a **False Negative** (failing to predict a stroke in a high-risk patient) is extremely high.

The **Gaussian Naive Bayes (GNB)** classifier was chosen for its computational efficiency, simplicity, and effectiveness on numerical features, particularly when the underlying feature distributions are assumed to be Gaussian.

## Key Files

| File Name | Description |
| :--- | :--- |
| `Stroke prediction_GaussianNB.py` | The main Jupyter Notebook containing the data preprocessing, Gaussian Naive Bayes model training, and performance evaluation. |
| `data-10.csv` | The dataset containing patient records with health factors and the binary stroke outcome. |
| `README.md` | This overview file. |

## Methodology

### 1. Data Preparation and Handling Imbalance
The dataset presented significant challenges, typical of medical data:

* **Missing Values:** Missing values in critical features like `bmi` were addressed through imputation (e.g., using the mean).
* **Categorical Encoding:** Features like `gender`, `work_type`, and `smoking_status` were converted into numerical formats using techniques such as Label Encoding.
* **Addressing Class Imbalance:** The stroke class (positive cases) is a severe minority. This inherent imbalance was likely addressed (or needs future attention) to prevent the model from becoming biased towards the dominant 'No Stroke' class.

### 2. Gaussian Naive Bayes Classifier
The GNB model was trained to predict the probability of stroke:

* **Assumption:** The model assumes that the features within each class (Stroke/No Stroke) follow a normal (Gaussian) distribution and that features are conditionally independent.
* **Model Training:** The classifier calculates the mean and variance of each feature for both classes to determine the likelihood of a patient belonging to the 'Stroke' class.

## Model Performance Analysis

The model achieved high overall accuracy, but exhibited classic issues associated with class imbalance, requiring careful interpretation of the minority class metrics:

| Metric | Overall Result |
| :--- | :--- |
| **Accuracy** | $\mathbf{0.87}$ |
| **F1-Score (Weighted Avg)** | $\mathbf{0.89}$ |

### Class-Specific Analysis (Critical for Medical Applications)

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **No Stroke (0)** | $\mathbf{0.96}$ | $\mathbf{0.89}$ | $\mathbf{0.93}$ |
| **Stroke (1)** | $\mathbf{0.22}$ | $\mathbf{0.47}$ | $\mathbf{0.30}$ |

#### Interpretation:

1.  **Safety and Risk (Class 1 - Stroke):**
    * **Low Recall ($\mathbf{0.47}$):** This is the model's most critical weakness. It correctly identifies **less than half ($47\%$)** of the patients who will actually suffer a stroke. This results in an unacceptably high number of **False Negatives**, which is dangerous for patient care.
    * **Low Precision ($\mathbf{0.22}$):** When the model *predicts* a stroke, it is wrong $78\%$ of the time. This results in high **False Positives**, leading to unnecessary testing and patient anxiety, but is less critical than missed stroke cases.

2.  **Model Bias (Class 0 - No Stroke):**
    * The model performs extremely well on the majority class ($\mathbf{0.93}$ F1-Score), which inflates the overall $\mathbf{87\%}$ accuracy score.

**Conclusion:**

The Gaussian Naive Bayes model provides a high overall accuracy due to its strong performance on the 'No Stroke' majority class. However, its low **Recall** ($\mathbf{0.47}$) for the life-threatening 'Stroke' class makes the current model inadequate for clinical deployment. Future work must focus on **imbalance techniques** (e.g., SMOTE, adjusting class weights, or collecting more positive samples) to significantly increase the model's sensitivity in detecting true stroke risks.

## Technologies and Libraries

* **Python 3.x**
* **VS code**
* `pandas` & `numpy` (for data manipulation)
* `scikit-learn` (for **GaussianNB**, preprocessing, and metrics)

You can also view the notebook directly on GitHub or platforms like **Google Colab** without needing a local setup.

## 👩‍💻 Author
**Aditi K**  
CSE (AI & ML) | Stroke Prediction | Gaussian NB
