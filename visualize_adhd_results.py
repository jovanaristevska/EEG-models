import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re

# ============================================================
# STEP 1 — Parse the results.csv file
# ============================================================
# The CSV has this format:
# "0:INFO ... adhd/eval epoch: 0", " loss: 0.408", " acc: 0.873", ...
# We parse each row to extract epoch, type (eval/test), and metrics

def parse_results(filepath):
    df = pd.read_csv(filepath, header=None)
    
    eval_rows = []
    test_rows = []
    
    for _, row in df.iterrows():
        line = str(row[0])
        
        # Extract epoch number
        epoch_match = re.search(r'epoch:\s*(\d+)', line)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))
        
        # Extract metrics from remaining columns
        full_line = ','.join([str(row[i]) for i in range(len(row))])
        
        loss_match = re.search(r'loss:\s*([\d.]+)', full_line)
        acc_match = re.search(r'acc:\s*([\d.]+)', full_line)
        bal_acc_match = re.search(r'balanced_acc:\s*([\d.]+)', full_line)
        auroc_match = re.search(r'auroc:\s*([\d.]+)', full_line)
        auc_pr_match = re.search(r'auc_pr:\s*([\d.]+)', full_line)
        
        metrics = {
            'epoch': epoch,
            'loss': float(loss_match.group(1)) if loss_match else None,
            'acc': float(acc_match.group(1)) if acc_match else None,
            'balanced_acc': float(bal_acc_match.group(1)) if bal_acc_match else None,
            'auroc': float(auroc_match.group(1)) if auroc_match else None,
            'auc_pr': float(auc_pr_match.group(1)) if auc_pr_match else None,
        }
        
        if 'eval epoch' in line:
            eval_rows.append(metrics)
        elif 'test epoch' in line:
            test_rows.append(metrics)
    
    eval_df = pd.DataFrame(eval_rows).sort_values('epoch').reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).sort_values('epoch').reset_index(drop=True)
    
    return eval_df, test_df


# ============================================================
# STEP 2 — Load data
# ============================================================
eval_df, test_df = parse_results('adhd_results.csv')

print("=== VALIDATION RESULTS ===")
print(eval_df.to_string(index=False))
print("\n=== TEST RESULTS ===")
print(test_df.to_string(index=False))

# Best epoch by Test AUROC
if not test_df.empty:
    best_idx = test_df['auroc'].idxmax()
    best = test_df.loc[best_idx]
    print(f"\n=== BEST TEST EPOCH ===")
    print(f"Epoch:         {int(best['epoch'])}")
    print(f"Test AUROC:    {best['auroc']:.4f}")
    print(f"Test ACC:      {best['acc']:.4f}")
    print(f"Test Bal ACC:  {best['balanced_acc']:.4f}")
    print(f"Test AUC-PR:   {best['auc_pr']:.4f}")


# ============================================================
# STEP 3 — Plot
# ============================================================
plt.style.use('seaborn-v0_8-darkgrid')

fig = plt.figure(figsize=(16, 12))
fig.suptitle('EEGPT Model — ADHD Classification Results', 
             fontsize=18, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# Color scheme
VAL_COLOR = '#2196F3'   # blue for validation
TEST_COLOR = '#F44336'  # red for test

# ----------------------------------------------------------
# Plot 1 — AUROC
# ----------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(eval_df['epoch'], eval_df['auroc'], 
         color=VAL_COLOR, linewidth=2.5, marker='o', markersize=4, label='Validation')
if not test_df.empty:
    ax1.plot(test_df['epoch'], test_df['auroc'], 
             color=TEST_COLOR, linewidth=2.5, marker='s', markersize=4, label='Test')
    # Mark best test epoch
    ax1.axvline(x=best['epoch'], color='green', linestyle='--', alpha=0.6, label=f'Best epoch ({int(best["epoch"])})')
ax1.set_title('AUROC over Epochs', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('AUROC')
ax1.legend()
ax1.set_ylim([0.5, 1.0])

# ----------------------------------------------------------
# Plot 2 — Accuracy
# ----------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(eval_df['epoch'], eval_df['acc'], 
         color=VAL_COLOR, linewidth=2.5, marker='o', markersize=4, label='Val Accuracy')
ax2.plot(eval_df['epoch'], eval_df['balanced_acc'], 
         color=VAL_COLOR, linewidth=2, linestyle='--', marker='o', markersize=4, label='Val Balanced Acc', alpha=0.7)
if not test_df.empty:
    ax2.plot(test_df['epoch'], test_df['acc'], 
             color=TEST_COLOR, linewidth=2.5, marker='s', markersize=4, label='Test Accuracy')
    ax2.plot(test_df['epoch'], test_df['balanced_acc'], 
             color=TEST_COLOR, linewidth=2, linestyle='--', marker='s', markersize=4, label='Test Balanced Acc', alpha=0.7)
ax2.set_title('Accuracy over Epochs', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend(fontsize=8)
ax2.set_ylim([0.5, 1.0])

# ----------------------------------------------------------
# Plot 3 — Loss
# ----------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(eval_df['epoch'], eval_df['loss'], 
         color=VAL_COLOR, linewidth=2.5, marker='o', markersize=4, label='Validation Loss')
if not test_df.empty:
    ax3.plot(test_df['epoch'], test_df['loss'], 
             color=TEST_COLOR, linewidth=2.5, marker='s', markersize=4, label='Test Loss')
ax3.set_title('Loss over Epochs', fontsize=13, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend()

# ----------------------------------------------------------
# Plot 4 — Final metrics bar chart
# ----------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])

if not test_df.empty:
    metrics_labels = ['AUROC', 'Accuracy', 'Balanced Acc', 'AUC-PR']
    
    # Best test epoch values
    best_test_vals = [
        best['auroc'],
        best['acc'],
        best['balanced_acc'],
        best['auc_pr'],
    ]
    
    # Final epoch values
    final_test = test_df.iloc[-1]
    final_test_vals = [
        final_test['auroc'],
        final_test['acc'],
        final_test['balanced_acc'],
        final_test['auc_pr'],
    ]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, best_test_vals, width, 
                    label=f'Best Epoch ({int(best["epoch"])})', 
                    color='#4CAF50', alpha=0.85, edgecolor='white')
    bars2 = ax4.bar(x + width/2, final_test_vals, width, 
                    label=f'Final Epoch ({int(final_test["epoch"])})', 
                    color=TEST_COLOR, alpha=0.85, edgecolor='white')
    
    # Add value labels on bars
    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax4.set_title('Final Test Metrics Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_labels)
    ax4.set_ylabel('Score')
    ax4.set_ylim([0, 1.15])
    ax4.legend()

plt.savefig('adhd_results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n✅ Plot saved as: adhd_results.png")
plt.show()