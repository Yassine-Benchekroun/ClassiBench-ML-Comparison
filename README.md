# ML Comparison Project

This project evaluates and compares different classification algorithms to predict user purchase behavior based on social network advertisement data.

## Project Structure
```text
ML-Comparison-Project/
├── data/               # Datasets (Social_Network_Data.xlsx)
├── src/                # Python scripts for model training (main.py)
├── results/            # Placeholder for results
       └── image.png
├── .gitignore          # Files Git should ignore
├── README.md           # Summary of model performance
└── requirements.txt    # Python dependencies
```

## Models Evaluated
- **Logistic Regression**: A baseline linear model for binary classification.
- **SVM (Linear Kernel)**: Support Vector Machine with a linear boundary.
- **SVM (Polynomial Kernel)**: SVM using a polynomial kernel to capture non-linear relationships.
- **SVM (RBF Kernel)**: SVM using a Radial Basis Function (Exponential) kernel for complex non-linear boundaries.

## Evaluation Metrics
The models are compared using the following metrics:
- **Accuracy**: Overall correctness of the model.
- **ROC AUC**: Ability of the model to distinguish between classes.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **Sensitivity (Recall)**: Ability to find all positive instances.
- **Specificity**: Ability to find all negative instances.

## How to Run
1. **Clone the repository** (or download the source).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python src/main.py
   ```
---
## Results
When executed, the script outputs a consolidated table of metrics directly to the console, sorted by **ROC AUC** in descending order. This provides a quick and clear comparison of which classification method performed best on the dataset.
---
![Comparison of SVM and Logistic Regression](./result/Result.jpeg)
