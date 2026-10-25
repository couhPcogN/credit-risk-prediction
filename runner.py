from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from model import KNN, LinearRegression, LogisticRegression


BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "web_app" / "prediction" / "static" / "images"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_manually(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return float(acc), float(prec), float(rec), float(f1)

def plot_confusion_matrix(y_true, y_pred, model_name):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    
    cm = np.array([[TN, FP],
                   [FN, TP]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.6)
    fig.colorbar(cax)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Safe (0)', 'Default (1)'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Safe (0)', 'Default (1)'])
    ax.xaxis.set_ticks_position('bottom')
    
    labels_text = [
        [f"TN = {TN}\n(Correctly predicted\nSafe)", f"FP = {FP}\n(False alarm\nfor Default)"],
        [f"FN = {FN}\n(Missed Default\n- HIGH RISK)", f"TP = {TP}\n(Correctly predicted\nDefault)"]
    ]
    
    for i in range(2):
        for j in range(2):
            ax.text(x=j, y=i, s=labels_text[i][j], 
                    va='center', ha='center', fontsize=11, fontweight='bold', color='black')
            
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=13, pad=15)
    ax.set_xlabel('Predicted Label', fontsize=11, labelpad=10)
    ax.set_ylabel('Actual Label', fontsize=11, labelpad=10)
    
    plt.tight_layout()
    filename = f'confusion_matrix_{model_name.lower()}.png'
    plt.savefig(REPORT_DIR / filename, dpi=300)
    plt.close()
    print(f"[SUCCESS] Confusion matrix saved to '{filename}'.")

def calculate_roc_auc_manually(y_true, y_scores):
    desc_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_indices]
    
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    
    tp_cumsum = np.cumsum(y_true_sorted == 1)
    fp_cumsum = np.cumsum(y_true_sorted == 0)
    
    tpr = tp_cumsum / P
    fpr = fp_cumsum / N
    
    tpr = np.insert(tpr, 0, 0.0)
    fpr = np.insert(fpr, 0, 0.0)
    
    try:
        auc = np.trapezoid(tpr, fpr)
    except AttributeError:
        auc = np.trapz(tpr, fpr)
        
    return fpr, tpr, auc

def plot_roc_curve_comparison(y_true, scores_dict):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'r--', label='Random Guess (AUC = 0.500)')
    
    colors = {'KNN': '#3498db', 'Linear Regression': '#e74c3c', 'Logistic Regression': '#2ecc71'}
    
    for model_name, y_scores in scores_dict.items():
        fpr, tpr, auc = calculate_roc_auc_manually(y_true, y_scores)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', color=colors[model_name], lw=2.5)
        
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve and AUC (20% Test Set)', fontsize=14, pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'roc_curve_comparison.png', dpi=300)
    plt.close()
    print("[SUCCESS] ROC curve saved to 'roc_curve_comparison.png'.")

def plot_model_comparison(results_dict):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    knn_scores = results_dict['KNN']
    lr_scores = results_dict['Linear Regression']
    log_scores = results_dict['Logistic Regression']

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, knn_scores, width, label='KNN', color='#3498db')
    rects2 = ax.bar(x, lr_scores, width, label='Linear Regression', color='#e74c3c')
    rects3 = ax.bar(x + width, log_scores, width, label='Logistic Regression', color='#2ecc71')

    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison (80% Train - 20% Test)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right')

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    filename = 'comparison_80_20_chart.png'
    plt.savefig(REPORT_DIR / filename, dpi=300)
    plt.close()
    print(f"[SUCCESS] Model comparison chart saved to '{filename}'.")

def plot_feature_importance(coefs, model_name):
    importance = np.abs(coefs)
    
    feature_names = [
        'person_age', 'person_income', 'person_home_ownership', 
        'person_emp_length', 'loan_intent', 'loan_grade', 
        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
        'cb_person_default_on_file', 'cb_person_cred_hist_length'
    ]
    
    if len(importance) != len(feature_names):
        feature_names = [f'Feature {i+1}' for i in range(len(importance))]
        
    sorted_indices = np.argsort(importance)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(importance)), importance[sorted_indices], color='#2c3e50', alpha=0.9)
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([feature_names[i] for i in sorted_indices], fontsize=10)
    ax.set_xlabel('Feature Importance (Absolute Coefficient)', fontsize=11)
    ax.set_title(f'Feature Importance - {model_name}', fontsize=13, pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    filename = 'feature_importance_logistic.png'
    plt.savefig(REPORT_DIR / filename, dpi=300)
    plt.close()
    print(f"[SUCCESS] Feature-importance chart saved to '{filename}'.")

if __name__ == "__main__":
    print("Loading credit-risk data...")
    
    X_train = np.load(BASE_DIR / 'X_train.npy')
    y_train = np.load(BASE_DIR / 'y_train.npy')
    X_test = np.load(BASE_DIR / 'X_test.npy')
    y_test = np.load(BASE_DIR / 'y_test.npy')

    print("\n--- TRAINING AND EVALUATING MODELS ---")
    
    y_scores_dict = {}
    results_dict = {}
    
    print("\nTraining KNN (k=5)...")
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    
    probs_knn = []
    for x in X_test:
        distances = np.sqrt(np.sum((knn.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:knn.k]
        k_nearest_labels = knn.y_train[k_indices]
        probs_knn.append(np.sum(k_nearest_labels == 1) / knn.k)
        
    y_scores_dict['KNN'] = np.array(probs_knn)
    results_dict['KNN'] = evaluate_manually(y_test, y_pred_knn)
    plot_confusion_matrix(y_test, y_pred_knn, 'KNN')

    print("\nTraining Linear Regression...")
    lr = LinearRegression(lr=0.1, iterations=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    y_scores_dict['Linear Regression'] = np.dot(X_test, lr.weights) + lr.bias
    results_dict['Linear Regression'] = evaluate_manually(y_test, y_pred_lr)
    plot_confusion_matrix(y_test, y_pred_lr, 'Linear')

    print("\nTraining Logistic Regression...")
    log_reg = LogisticRegression(lr=0.1, max_iterations=5000)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    
    y_scores_dict['Logistic Regression'] = log_reg.predict_prob(X_test)
    results_dict['Logistic Regression'] = evaluate_manually(y_test, y_pred_log)
    plot_confusion_matrix(y_test, y_pred_log, 'Logistic')

    print("\n" + "="*80)
    print("DETAILED EVALUATION RESULTS (80% TRAIN - 20% TEST)")
    print("="*80)
    
    for m in ['KNN', 'Linear Regression', 'Logistic Regression']:
        acc, prec, rec, f1 = results_dict[m]
        print(f"Model {m.upper()}:")
        print(f"  Accuracy  : {acc:.16f}")
        print(f"  Precision : {prec:.16f}")
        print(f"  Recall    : {rec:.16f}")
        print(f"  F1-Score  : {f1:.16f}")
        print("-" * 80)

    print("\nGenerating analysis reports...")
    plot_model_comparison(results_dict)
    plot_roc_curve_comparison(y_test, y_scores_dict)
    
    plot_feature_importance(log_reg.theta, 'Logistic Regression')
    
    print("\n[COMPLETE] All report images and evaluation results are ready.")