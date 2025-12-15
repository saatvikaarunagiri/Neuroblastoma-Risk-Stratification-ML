# Neuroblastoma Risk Stratification ML Pipeline

Machine learning pipeline for predicting pediatric cancer risk groups from clinical and genomic features.

## Getting Started

These instructions will help you run the neuroblastoma risk prediction pipeline.

### Prerequisites

Before running this project, you need:

* Python 3.8 or higher
* 8GB RAM minimum
* pip package manager
* Command line access

## Usage

### Running the Complete Pipeline

```
$ python pipeline.py
```

The pipeline executes:
* Data generation (1,200 synthetic patient samples)
* Exploratory data analysis
* Data preprocessing and SMOTE balancing
* Training 4 machine learning models
* Model evaluation and comparison
* Survival analysis validation
* Output generation

### Expected Runtime

* Complete pipeline: 2-3 minutes
* Generates 9 figures automatically

## Results

### Model Performance

* Best Model: Gradient Boosting
* Accuracy: 92.1%
* F1-Score: 0.917
* Mean AUC: 0.94

### Feature Importance

1. MYCN amplification (0.35)
2. Age at diagnosis (0.22)
3. Disease stage (0.18)

### Clinical Validation

* Survival analysis: Significant stratification (log-rank p<0.001)
* Cox regression: MYCN HR=3.2, Age HR=1.05

## Output Files

The pipeline generates:

```
outputs/
├── neuroblastoma_best_model.pkl
├── neuroblastoma_scaler.pkl
├── neuroblastoma_label_encoder.pkl
├── neuroblastoma_predictions.csv
├── neuroblastoma_performance_report.txt
└── figures/
    ├── risk_group_distribution.png
    ├── feature_correlation.png
    ├── mycn_by_risk.png
    ├── age_distributions.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── roc_curves.png
    └── survival_curves.png
```

## Technical Details

* Dataset: 1,200 patients (synthetic, realistic correlations)
* Features: 8 (5 clinical, 3 genomic)
* Target: 4 risk groups (Low, Intermediate, High, Very High)
* Class balance: SMOTE oversampling
* Cross-validation: 5-fold stratified
* Models tested: Gradient Boosting, Random Forest, XGBoost, Logistic Regression

## Clinical Application

Risk groups guide treatment intensity:
* Low Risk: Observation only (>95% survival)
* Intermediate: Moderate chemotherapy (~80% survival)
* High Risk: Intensive treatment (~60% survival)
* Very High: Maximal therapy (~30% survival)

## Additional Information

* Application: Pediatric oncology treatment planning
* Target cancer: Neuroblastoma (most common extracranial solid tumor in children)
* Guidelines: International Neuroblastoma Risk Group (INRG) classification
