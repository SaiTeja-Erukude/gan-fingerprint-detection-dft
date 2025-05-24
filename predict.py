import os
import csv
import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt
from keras.models               import load_model
from keras.preprocessing.image  import ImageDataGenerator
from sklearn.metrics            import confusion_matrix, roc_curve, auc, precision_recall_curve
from datetime                   import datetime

# Configuration
MODEL_PATH     = "models/best_model.h5"
TEST_DATA_PATH = "data/fourier/test"
HEIGHT, WIDTH  = 256, 256
BATCH_SIZE     = 32

# Create output directory for metrics
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
METRICS_DIR = f"predictions/{timestamp}"
os.makedirs(METRICS_DIR, exist_ok=True)

# Load the saved model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Create test data generator (no augmentation, just rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test data
test_generator = test_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Important for keeping correct order for metrics
)

# Get class mappings
class_indices = test_generator.class_indices
class_names = list(class_indices.keys())
print(f"Class indices: {class_indices}")

# Save class mapping to CSV
with open(os.path.join(METRICS_DIR, 'class_mapping.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class Index', 'Class Name'])
    for name, idx in class_indices.items():
        writer.writerow([idx, name])

# Evaluate model on test data
print("Evaluating model...")
evaluation_results = model.evaluate(test_generator)

# If evaluation_results is a single value, convert to list
if not isinstance(evaluation_results, list):
    evaluation_results = [evaluation_results]

# Get metric names from model
if hasattr(model, 'metrics_names'):
    metric_names = model.metrics_names
else:
    metric_names = ['loss'] + [f'metric_{i}' for i in range(len(evaluation_results)-1)]

# Ensure we have enough names for all results
while len(metric_names) < len(evaluation_results):
    metric_names.append(f'metric_{len(metric_names)}')

# Map results to their names
metrics_dict = dict(zip(metric_names, evaluation_results))

# Get loss and accuracy (if available)
test_loss = metrics_dict.get('loss', evaluation_results[0])
test_acc = metrics_dict.get('accuracy', metrics_dict.get('acc', None))

# Print available metrics
print("Model evaluation results:")
for name, value in metrics_dict.items():
    print(f"{name}: {value:.4f}")

# Save all metrics to CSV
with open(os.path.join(METRICS_DIR, 'basic_metrics.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Metric', 'Value'])
    for name, value in metrics_dict.items():
        writer.writerow([name, value])

# Generate predictions
print("Generating predictions...")
test_generator.reset()  # Reset generator to ensure we get all samples
y_pred_prob = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred_prob, axis=1)
y_true = test_generator.classes

# Save individual predictions to CSV
predictions_df = pd.DataFrame({
    'True Class': [class_names[i] for i in y_true],
    'Predicted Class': [class_names[i] for i in y_pred_classes],
    'Correct': y_true == y_pred_classes
})

# Add probability columns for each class
for i, class_name in enumerate(class_names):
    predictions_df[f'{class_name}_Probability'] = y_pred_prob[:, i]

# Calculate metrics
print("\nGenerating metrics...")

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Save confusion matrix to CSV
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(METRICS_DIR, 'confusion_matrix.csv'))

# 2. ROC Curve & AUC data
def calculate_save_roc(y_true, y_scores, class_names):
    plt.figure(figsize=(10, 8))
    
    # Convert to one-hot encoding for ROC curve calculation
    y_true_binary = np.zeros((len(y_true), len(class_names)))
    for i, label in enumerate(y_true):
        y_true_binary[i, label] = 1
    
    # Calculate ROC curve and AUC for each class and save data
    roc_data = {}
    
    for i, class_name in enumerate(class_names):
        try:
            fpr, tpr, thresholds = roc_curve(y_true_binary[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            # Save ROC data - creating separate DataFrames to avoid different length arrays
            fpr_tpr_df = pd.DataFrame({
                'False Positive Rate': fpr,
                'True Positive Rate': tpr
            })
            fpr_tpr_df.to_csv(os.path.join(METRICS_DIR, f'roc_curve_{class_name}_fpr_tpr.csv'), index=False)
            
            # Save thresholds separately if they exist
            if len(thresholds) > 0:
                thresholds_df = pd.DataFrame({
                    'Thresholds': thresholds
                })
                thresholds_df.to_csv(os.path.join(METRICS_DIR, f'roc_curve_{class_name}_thresholds.csv'), index=False)
            
            # Store AUC value
            roc_data[class_name] = roc_auc
            
            # Plot
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
        except Exception as e:
            print(f"Error calculating ROC for class {class_name}: {e}")
            roc_data[class_name] = float('nan')
    
    # Save AUC values to CSV
    auc_df = pd.DataFrame(list(roc_data.items()), columns=['Class', 'AUC'])
    auc_df.to_csv(os.path.join(METRICS_DIR, 'auc_values.csv'), index=False)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(METRICS_DIR, 'roc_curve.png'))
    plt.close()
    
    return roc_data

# 3. Precision-Recall Curve data
def calculate_save_precision_recall(y_true, y_scores, class_names):
    plt.figure(figsize=(10, 8))
    
    # Convert to one-hot encoding
    y_true_binary = np.zeros((len(y_true), len(class_names)))
    for i, label in enumerate(y_true):
        y_true_binary[i, label] = 1
    
    pr_data = {}
    
    # Calculate precision-recall curve for each class
    for i, class_name in enumerate(class_names):
        try:
            precision, recall, thresholds = precision_recall_curve(y_true_binary[:, i], y_pred_prob[:, i])
            avg_precision = np.mean(precision)
            
            # Save PR data - creating separate DataFrames to avoid different length arrays
            precision_recall_df = pd.DataFrame({
                'Precision': precision,
                'Recall': recall
            })
            precision_recall_df.to_csv(os.path.join(METRICS_DIR, f'precision_recall_{class_name}_values.csv'), index=False)
            
            # Save thresholds separately if they exist
            if len(thresholds) > 0:
                thresholds_df = pd.DataFrame({
                    'Thresholds': thresholds
                })
                thresholds_df.to_csv(os.path.join(METRICS_DIR, f'precision_recall_{class_name}_thresholds.csv'), index=False)
            
            # Store average precision
            pr_data[class_name] = avg_precision
            
            # Plot
            plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {avg_precision:.2f})')
        except Exception as e:
            print(f"Error calculating Precision-Recall for class {class_name}: {e}")
            pr_data[class_name] = float('nan')
    
    # Save average precision values to CSV
    ap_df = pd.DataFrame(list(pr_data.items()), columns=['Class', 'Average Precision'])
    ap_df.to_csv(os.path.join(METRICS_DIR, 'avg_precision.csv'), index=False)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(os.path.join(METRICS_DIR, 'precision_recall_curve.png'))
    plt.close()
    
    return pr_data

# 4. Plot Confusion Matrix as heatmap
def plot_save_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    
    # If seaborn is available, use it for nicer visualization
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
    except ImportError:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(METRICS_DIR, 'confusion_matrix.png'))
    plt.close()

# Create a function to handle potential errors
def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error running {func.__name__}: {e}")
        return {}

# Generate all metrics and visualizations with error handling
print("Calculating and saving ROC curves...")
roc_data = safe_run(calculate_save_roc, y_true, y_pred_prob, class_names)

print("Calculating and saving precision-recall curves...")
pr_data = safe_run(calculate_save_precision_recall, y_true, y_pred_prob, class_names)

print("Saving confusion matrix visualization...")
safe_run(plot_save_confusion_matrix, cm, class_names)

# Function to create a summary metrics file
def save_summary_metrics():
    try:
        # Calculate overall metrics
        overall_accuracy = np.mean(y_true == y_pred_classes)
        
        # Create summary dictionary
        summary = {
            "model_path": MODEL_PATH,
            "test_data_path": TEST_DATA_PATH,
            "metrics": {k: float(v) for k, v in metrics_dict.items()},
            "overall_accuracy": float(overall_accuracy),
            "number_of_test_samples": int(len(y_true)),
            "class_distribution": {class_name: int(np.sum(y_true == idx)) 
                                for class_name, idx in class_indices.items()},
        }
        
        # Add ROC data if available
        if roc_data:
            summary["roc_auc_scores"] = {k: float(v) for k, v in roc_data.items()}
            
        # Add PR data if available
        if pr_data:
            summary["average_precision_scores"] = {k: float(v) for k, v in pr_data.items()}
            
        summary["timestamp"] = timestamp
        
        # Save as CSV
        with open(os.path.join(METRICS_DIR, 'summary_metrics.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Value'])
            
            # Add all metrics
            for name, value in metrics_dict.items():
                writer.writerow([name, value])
            
            writer.writerow(['Overall Accuracy', overall_accuracy])
            writer.writerow(['Number of Test Samples', len(y_true)])
            
            # Add class distribution
            for class_name, idx in class_indices.items():
                writer.writerow([f'Class {class_name} Count', np.sum(y_true == idx)])
            
            # Add ROC AUC scores if available
            if roc_data:
                for class_name, score in roc_data.items():
                    writer.writerow([f'ROC AUC ({class_name})', score])
            
            # Add Average Precision scores if available
            if pr_data:
                for class_name, score in pr_data.items():
                    writer.writerow([f'Average Precision ({class_name})', score])
    except Exception as e:
        print(f"Error saving summary metrics: {e}")

# Save summary metrics
safe_run(save_summary_metrics)

print(f"All metrics have been saved to the '{METRICS_DIR}' directory.")
print(f"Files saved:")
for file in os.listdir(METRICS_DIR):
    print(f" - {file}")