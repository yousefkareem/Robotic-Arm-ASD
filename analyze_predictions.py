import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, roc_curve

def analyze_predictions(machine_type='RoboticArm'):
    # Read labels
    labels_df = pd.read_csv('labels/eval_labels.tsv', sep='\t', header=None, names=['filename', 'label'])
    labels_df['label'] = labels_df['label'].map({'normal': 0, 'anomaly': 1, 'unknown': -1})
    labels_df = labels_df[labels_df['label'] != -1]  # Remove unknown labels
    
    # Find score files
    score_files = glob(f'results/eval_data/baseline*/anomaly_score_DCASE2024T2{machine_type}_*.csv')
    
    if not score_files:
        print(f"No score files found for {machine_type}")
        return
    
    print(f"Found {len(score_files)} score files:")
    for f in score_files:
        print(f"  - {f}")
    
    # Read and merge scores
    scores_df = pd.concat([pd.read_csv(f, header=None, names=['filename', 'score']) for f in score_files])
    merged_df = pd.merge(scores_df, labels_df, on='filename', how='inner')
    
    print("\n" + "="*50)
    print("Overall Analysis")
    print("="*50)
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"Total samples: {len(merged_df)}")
    print(f"Normal samples: {len(merged_df[merged_df['label'] == 0])}")
    print(f"Anomaly samples: {len(merged_df[merged_df['label'] == 1])}")
    
    # Score statistics by class
    print("\nScore Statistics by Class:")
    print(merged_df.groupby('label')['score'].describe())
    
    # Calculate metrics
    auc = roc_auc_score(merged_df['label'], merged_df['score'])
    ap = average_precision_score(merged_df['label'], merged_df['score'])
    
    # Calculate pAUC (partial AUC up to 0.1 FPR)
    fpr, tpr, _ = roc_curve(merged_df['label'], merged_df['score'])
    p_auc = np.trapz(tpr[fpr <= 0.1], fpr[fpr <= 0.1]) / 0.1
    
    print("\nPerformance Metrics:")
    print(f"AUC: {auc:.4f}")
    print(f"pAUC: {p_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    # Calculate confusion matrix at different thresholds
    thresholds = np.percentile(merged_df['score'], [25, 50, 75, 90, 95, 99])
    print("\nConfusion Matrices at Different Thresholds:")
    for threshold in thresholds:
        predictions = (merged_df['score'] >= threshold).astype(int)
        tn = np.sum((predictions == 0) & (merged_df['label'] == 0))
        fp = np.sum((predictions == 1) & (merged_df['label'] == 0))
        fn = np.sum((predictions == 0) & (merged_df['label'] == 1))
        tp = np.sum((predictions == 1) & (merged_df['label'] == 1))
        
        print(f"\nThreshold: {threshold:.4f}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        print(f"Precision: {tp/(tp+fp):.4f}")
        print(f"Recall: {tp/(tp+fn):.4f}")
        print(f"F1 Score: {2*tp/(2*tp+fp+fn):.4f}")
    
    # Plot score distributions
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data=merged_df, x='score', hue='label', bins=50, kde=True)
    plt.title('Score Distribution by Class')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    
    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=merged_df, x='label', y='score')
    plt.title('Score Distribution by Class')
    plt.xlabel('Label (0=Normal, 1=Anomaly)')
    plt.ylabel('Anomaly Score')
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/score_analysis.png')
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png')
    plt.close()

if __name__ == "__main__":
    analyze_predictions() 