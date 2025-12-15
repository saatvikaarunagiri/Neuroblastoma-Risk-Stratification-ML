#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, roc_auc_score, roc_curve,
                             precision_recall_curve, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

#random seed for reproducibility
np.random.seed(42)

#DATA GENERATION
def generate_neuroblastoma_data(n_samples=1200):
    
    data = []
    
    #Risk group distribution 
    risk_dist = {
        'Low': 0.30,       
        'Intermediate': 0.25,  
        'High': 0.30,      
        'Very High': 0.15  
    }
    
    for risk_group in risk_dist.keys():
        n_group = int(n_samples * risk_dist[risk_group])
        
        if risk_group == 'Low':
            #Low risk: young age, no MYCN amp, favorable biology
            age = np.random.normal(18, 8, n_group).clip(0, 60)  #months
            mycn = np.random.choice([0, 1], n_group, p=[0.95, 0.05])
            del_11q = np.random.choice([0, 1], n_group, p=[0.90, 0.10])
            ploidy = np.random.choice([0, 1], n_group, p=[0.30, 0.70])  #mostly hyperdiploid
            stage = np.random.choice([1, 2, 3, 4], n_group, p=[0.4, 0.3, 0.2, 0.1])
            ldh = np.random.normal(400, 100, n_group).clip(200, 1000)
            ferritin = np.random.normal(100, 50, n_group).clip(20, 300)
            tumor_size = np.random.normal(5, 2, n_group).clip(2, 12)
            
        elif risk_group == 'Intermediate':
            #Intermediate risk: moderate age, mixed features
            age = np.random.normal(24, 12, n_group).clip(0, 72)
            mycn = np.random.choice([0, 1], n_group, p=[0.85, 0.15])
            del_11q = np.random.choice([0, 1], n_group, p=[0.70, 0.30])
            ploidy = np.random.choice([0, 1], n_group, p=[0.50, 0.50])
            stage = np.random.choice([2, 3, 4], n_group, p=[0.3, 0.5, 0.2])
            ldh = np.random.normal(800, 200, n_group).clip(300, 2000)
            ferritin = np.random.normal(200, 80, n_group).clip(50, 500)
            tumor_size = np.random.normal(8, 3, n_group).clip(3, 15)
            
        elif risk_group == 'High':
            #High risk: older age, some MYCN amp, unfavorable biology
            age = np.random.normal(36, 15, n_group).clip(12, 120)
            mycn = np.random.choice([0, 1], n_group, p=[0.60, 0.40])
            del_11q = np.random.choice([0, 1], n_group, p=[0.50, 0.50])
            ploidy = np.random.choice([0, 1], n_group, p=[0.70, 0.30])  #mostly diploid
            stage = np.random.choice([3, 4], n_group, p=[0.3, 0.7])
            ldh = np.random.normal(1500, 300, n_group).clip(500, 5000)
            ferritin = np.random.normal(400, 150, n_group).clip(100, 1000)
            tumor_size = np.random.normal(12, 4, n_group).clip(5, 20)
            
        else:  #Very High
            #Very high risk: MYCN amplified, stage 4, very unfavorable
            age = np.random.normal(48, 18, n_group).clip(18, 180)
            mycn = np.random.choice([0, 1], n_group, p=[0.20, 0.80])  #mostly amplified
            del_11q = np.random.choice([0, 1], n_group, p=[0.40, 0.60])
            ploidy = np.random.choice([0, 1], n_group, p=[0.85, 0.15])  #mostly diploid
            stage = np.random.choice([4], n_group)  #all stage 4
            ldh = np.random.normal(2500, 500, n_group).clip(1000, 8000)
            ferritin = np.random.normal(700, 200, n_group).clip(200, 2000)
            tumor_size = np.random.normal(15, 5, n_group).clip(8, 25)
        
        #Create dataframe for this risk group
        group_df = pd.DataFrame({
            'Age_months': age,
            'MYCN_amplified': mycn,
            '11q_deletion': del_11q,
            'Hyperdiploid': ploidy,
            'Stage': stage,
            'LDH': ldh,
            'Ferritin': ferritin,
            'Tumor_size_cm': tumor_size,
            'Risk_Group': risk_group
        })
        
        data.append(group_df)
    
    #Combine all risk groups
    df = pd.concat(data, ignore_index=True)
    
    #Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    #Generate survival data (correlated with risk)
    survival_months = []
    event_occurred = []
    
    for _, row in df.iterrows():
        if row['Risk_Group'] == 'Low':
            #Excellent prognosis
            surv = np.random.exponential(120, 1)[0]  #mean 10 years
            event = np.random.choice([0, 1], p=[0.95, 0.05])  #95% survival
        elif row['Risk_Group'] == 'Intermediate':
            surv = np.random.exponential(80, 1)[0]  #mean 6.7 years
            event = np.random.choice([0, 1], p=[0.80, 0.20])  #80% survival
        elif row['Risk_Group'] == 'High':
            surv = np.random.exponential(48, 1)[0]  #mean 4 years
            event = np.random.choice([0, 1], p=[0.60, 0.40])  #60% survival
        else:  #Very High
            surv = np.random.exponential(24, 1)[0]  #mean 2 years
            event = np.random.choice([0, 1], p=[0.30, 0.70])  #30% survival
        
        survival_months.append(min(surv, 120))  #cap at 10 years follow-up
        event_occurred.append(event)
    
    df['Survival_months'] = survival_months
    df['Event'] = event_occurred  #1=death, 0=censored
    
    return df

#Generate data
df = generate_neuroblastoma_data(n_samples=1200)

print(f" Generated {len(df)} patient records")
print(f"\nRisk Group Distribution:")
print(df['Risk_Group'].value_counts().sort_index())
print(f"\nFeature Summary:")
print(df.describe())

#EXPLORATORY DATA ANALYSIS
#Create output directory for figures
import os
os.makedirs('neuroblastoma_figures', exist_ok=True)

#Figure 1: Risk group distribution
plt.figure(figsize=(10, 6))
risk_counts = df['Risk_Group'].value_counts()
risk_order = ['Low', 'Intermediate', 'High', 'Very High']
colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
plt.bar(risk_order, [risk_counts[r] for r in risk_order], color=colors)
plt.xlabel('Risk Group', fontsize=12, fontweight='bold')
plt.ylabel('Number of Patients', fontsize=12, fontweight='bold')
plt.title('Neuroblastoma Risk Group Distribution (N=1200)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('neuroblastoma_figures/01_risk_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

#Figure 2: Feature correlation heatmap
plt.figure(figsize=(12, 10))
feature_cols = ['Age_months', 'MYCN_amplified', '11q_deletion', 'Hyperdiploid', 
                'Stage', 'LDH', 'Ferritin', 'Tumor_size_cm']
corr_matrix = df[feature_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('neuroblastoma_figures/02_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

#Figure 3: MYCN amplification by risk group
plt.figure(figsize=(10, 6))
mycn_risk = pd.crosstab(df['Risk_Group'], df['MYCN_amplified'], normalize='index') * 100
mycn_risk = mycn_risk.reindex(risk_order)
mycn_risk.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'])
plt.xlabel('Risk Group', fontsize=12, fontweight='bold')
plt.ylabel('Percentage of Patients', fontsize=12, fontweight='bold')
plt.title('MYCN Amplification Status by Risk Group', fontsize=14, fontweight='bold')
plt.legend(['Normal', 'Amplified'], title='MYCN Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('neuroblastoma_figures/03_mycn_by_risk.png', dpi=300, bbox_inches='tight')
plt.close()

#Figure 4: Age distribution by risk group
plt.figure(figsize=(12, 6))
for i, risk in enumerate(risk_order):
    plt.subplot(1, 4, i+1)
    data = df[df['Risk_Group'] == risk]['Age_months']
    plt.hist(data, bins=20, color=colors[i], alpha=0.7, edgecolor='black')
    plt.xlabel('Age (months)', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title(f'{risk} Risk\n(n={len(data)})', fontsize=11, fontweight='bold')
plt.suptitle('Age Distribution by Risk Group', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('neuroblastoma_figures/04_age_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

#DATA PREPROCESSING
#Separate features and target
X = df[['Age_months', 'MYCN_amplified', '11q_deletion', 'Hyperdiploid',
        'Stage', 'LDH', 'Ferritin', 'Tumor_size_cm']].copy()
y = df['Risk_Group'].copy()

#Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
risk_classes = le.classes_
print(f" Risk classes: {risk_classes}")

#Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f" Training set: {len(X_train)} patients")
print(f" Test set: {len(X_test)} patients")

#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f" After SMOTE: {len(X_train_balanced)} training samples")
print(f"  Class distribution: {np.bincount(y_train_balanced)}")

#MODEL TRAINING
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    #Train
    model.fit(X_train_balanced, y_train_balanced)
    
    #Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    #Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    #Cross-validation
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f" Accuracy: {accuracy:.3f}")
    print(f" F1-Score: {f1:.3f}")
    print(f" CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

#Select best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
print(f" Best model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.3f})")

#MODEL EVALUATION

y_pred_best = results[best_model_name]['y_pred']
y_pred_proba_best = results[best_model_name]['y_pred_proba']

#Classification report
print("\nClassification Report: (classification_report(y_test, y_pred_best, target_names=risk_classes)) ")

#Figure 5: Model comparison
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
f1_scores = [results[m]['f1_score'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='#e74c3c')

plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, model_names, rotation=15, ha='right')
plt.legend()
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('neuroblastoma_figures/05_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

#Figure 6: Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=risk_classes, yticklabels=risk_classes,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Risk Group', fontsize=12, fontweight='bold')
plt.ylabel('True Risk Group', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix - {best_model_name}\n(Accuracy: {results[best_model_name]["accuracy"]:.1%})', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('neuroblastoma_figures/06_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

#Figure 7: Feature importance 
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[indices], color='#9b59b6')
    plt.yticks(range(len(importances)), [X.columns[i] for i in indices])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('neuroblastoma_figures/07_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n Top 3 most important features:")
    for i in range(min(3, len(importances))):
        print(f"  {i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.3f}")
      
#ROC CURVES (One-vs-Rest)
from sklearn.preprocessing import label_binarize

#Binarize labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]

#Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba_best[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_proba_best[:, i])

#Figure 8: ROC curves
plt.figure(figsize=(10, 8))
colors_roc = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
for i, color in zip(range(n_classes), colors_roc):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{risk_classes[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title(f'ROC Curves - {best_model_name} (One-vs-Rest)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('neuroblastoma_figures/08_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print(f" Mean AUC across all classes: {np.mean(list(roc_auc.values())):.3f}")

#SURVIVAL ANALYSIS

#Add predictions to original dataframe
df_test = df.iloc[X_test.index].copy()
df_test['Predicted_Risk'] = le.inverse_transform(y_pred_best)
df_test['Correct_Prediction'] = (df_test['Risk_Group'] == df_test['Predicted_Risk'])

#Kaplan-Meier curves by true risk group
kmf = KaplanMeierFitter()
plt.figure(figsize=(12, 7))

for i, risk in enumerate(risk_order):
    mask = df_test['Risk_Group'] == risk
    if mask.sum() > 0:
        kmf.fit(df_test.loc[mask, 'Survival_months'], 
                df_test.loc[mask, 'Event'],
                label=risk)
        kmf.plot_survival_function(ci_show=True, color=colors[i], linewidth=2)

plt.xlabel('Time (months)', fontsize=12, fontweight='bold')
plt.ylabel('Survival Probability', fontsize=12, fontweight='bold')
plt.title('Kaplan-Meier Survival Curves by Risk Group', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('neuroblastoma_figures/09_kaplan_meier.png', dpi=300, bbox_inches='tight')
plt.close()

#Log-rank test
print("\nLog-rank tests (pairwise):")
for i in range(len(risk_order)):
    for j in range(i+1, len(risk_order)):
        mask1 = df_test['Risk_Group'] == risk_order[i]
        mask2 = df_test['Risk_Group'] == risk_order[j]
        
        if mask1.sum() > 0 and mask2.sum() > 0:
            result = logrank_test(
                df_test.loc[mask1, 'Survival_months'],
                df_test.loc[mask2, 'Survival_months'],
                df_test.loc[mask1, 'Event'],
                df_test.loc[mask2, 'Event']
            )
            print(f"  {risk_order[i]} vs {risk_order[j]}: p = {result.p_value:.4f}")

#Cox proportional hazards model
print("\nCox Proportional Hazards Model:")
cox_data = df_test[['Survival_months', 'Event', 'Age_months', 'MYCN_amplified', 
                     '11q_deletion', 'Hyperdiploid', 'Stage', 'LDH', 
                     'Ferritin', 'Tumor_size_cm']].copy()
cox_data = cox_data.dropna()

cph = CoxPHFitter()
cph.fit(cox_data, duration_col='Survival_months', event_col='Event')
print(cph.summary[['coef', 'exp(coef)', 'p']])

#SAVE RESULTS
#Save model
import joblib
joblib.dump(best_model, 'neuroblastoma_best_model.pkl')
joblib.dump(scaler, 'neuroblastoma_scaler.pkl')
joblib.dump(le, 'neuroblastoma_label_encoder.pkl')
print(" Saved model, scaler, and label encoder")

#Save predictions
predictions_df = pd.DataFrame({
    'Patient_ID': df_test.index,
    'True_Risk': df_test['Risk_Group'],
    'Predicted_Risk': df_test['Predicted_Risk'],
    'Correct': df_test['Correct_Prediction'],
    'Survival_months': df_test['Survival_months'],
    'Event': df_test['Event']
})
predictions_df.to_csv('neuroblastoma_predictions.csv', index=False)
print(" Saved predictions to CSV")

#Save performance metrics
with open('neuroblastoma_performance_report.txt', 'w') as f:
    
    f.write("DATASET INFORMATION:\n")
    f.write(f"Total patients: {len(df)}\n")
    f.write(f"Training set: {len(X_train)} patients\n")
    f.write(f"Test set: {len(X_test)} patients\n\n")
    
    f.write("RISK GROUP DISTRIBUTION:\n")
    f.write(str(df['Risk_Group'].value_counts().sort_index()) + "\n\n")
    
    f.write("MODEL COMPARISON:\n")
    for model_name, res in results.items():
        f.write(f"\n{model_name}:\n")
        f.write(f"  Accuracy: {res['accuracy']:.3f}\n")
        f.write(f"  F1-Score: {res['f1_score']:.3f}\n")
        f.write(f"  CV Score: {res['cv_mean']:.3f} ± {res['cv_std']:.3f}\n")
    
    f.write(f"\n\nBEST MODEL: {best_model_name}\n")
    f.write("DETAILED CLASSIFICATION REPORT:\n")
    f.write(classification_report(y_test, y_pred_best, target_names=risk_classes))
    
    f.write("\n\nCONFUSION MATRIX:\n")
    f.write(str(cm) + "\n")
    
    f.write("\n\nROC AUC SCORES (One-vs-Rest):\n")
    for i, risk in enumerate(risk_classes):
        f.write(f"  {risk}: {roc_auc[i]:.3f}\n")
    f.write(f"  Mean AUC: {np.mean(list(roc_auc.values())):.3f}\n")

print(" Saved performance report")

#SUMMARY
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['accuracy']:.1%}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.3f}")
print(f"Mean AUC: {np.mean(list(roc_auc.values())):.3f}")
