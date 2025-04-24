#!/usr/bin/env python3

"""
Evaluation Metrics for PanicSense Hybrid Neural Network Model
This module provides comprehensive evaluation metrics for assessing model performance
"""

import numpy as np
import pandas as pd
import torch
import json
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class for evaluating PanicSense model performance with comprehensive metrics"""
    
    def __init__(self, sentiment_labels=None):
        """
        Initialize the evaluator
        
        Args:
            sentiment_labels: List of sentiment label categories
        """
        self.sentiment_labels = sentiment_labels or ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.sentiment_labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.sentiment_labels)}
    
    def evaluate_predictions(
        self, 
        true_labels: List[str], 
        predicted_labels: List[str], 
        confidences: List[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions against ground truth
        
        Args:
            true_labels: List of ground truth sentiment labels
            predicted_labels: List of predicted sentiment labels
            confidences: List of confidence scores for predictions (optional)
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        # Convert string labels to indices if needed
        true_indices = [self.label_to_idx.get(label, 0) if isinstance(label, str) else label for label in true_labels]
        pred_indices = [self.label_to_idx.get(label, 0) if isinstance(label, str) else label for label in predicted_labels]
        
        # Calculate basic metrics
        accuracy = accuracy_score(true_indices, pred_indices)
        
        # Calculate per-class and weighted metrics
        precision_weighted = precision_score(true_indices, pred_indices, average='weighted', zero_division=0)
        recall_weighted = recall_score(true_indices, pred_indices, average='weighted', zero_division=0)
        f1_weighted = f1_score(true_indices, pred_indices, average='weighted', zero_division=0)
        
        precision_macro = precision_score(true_indices, pred_indices, average='macro', zero_division=0)
        recall_macro = recall_score(true_indices, pred_indices, average='macro', zero_division=0)
        f1_macro = f1_score(true_indices, pred_indices, average='macro', zero_division=0)
        
        # Per-class metrics
        classification_rep = classification_report(
            true_indices, pred_indices, 
            target_names=self.sentiment_labels, 
            output_dict=True, 
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_indices, pred_indices, labels=range(len(self.sentiment_labels)))
        
        # Normalize confusion matrix (row-wise)
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Replace NaNs with zeros
        
        # Calculate class distribution metrics
        true_class_distribution = Counter(true_labels)
        pred_class_distribution = Counter(predicted_labels)
        
        # Calculate error distribution and bias metrics
        error_indices = [i for i, (true, pred) in enumerate(zip(true_indices, pred_indices)) if true != pred]
        error_rate = len(error_indices) / len(true_indices) if true_indices else 0
        
        # Error analysis
        error_analysis = {}
        if error_indices:
            # For each true class, identify what it's being misclassified as
            for true_class in range(len(self.sentiment_labels)):
                misclassifications = {}
                for i in error_indices:
                    if true_indices[i] == true_class:
                        pred_class = pred_indices[i]
                        pred_label = self.idx_to_label[pred_class]
                        misclassifications[pred_label] = misclassifications.get(pred_label, 0) + 1
                
                if misclassifications:
                    true_label = self.idx_to_label[true_class]
                    error_analysis[true_label] = misclassifications
        
        # Calculate confidence metrics if provided
        confidence_metrics = {}
        if confidences:
            # Average confidence
            avg_confidence = np.mean(confidences)
            avg_confidence_correct = np.mean([conf for i, conf in enumerate(confidences) 
                                             if true_indices[i] == pred_indices[i]]) if confidences else 0
            avg_confidence_incorrect = np.mean([conf for i, conf in enumerate(confidences) 
                                               if true_indices[i] != pred_indices[i]]) if error_indices else 0
            
            # Confidence distribution
            confidence_distribution = {
                'min': min(confidences),
                'max': max(confidences),
                'mean': avg_confidence,
                'median': np.median(confidences),
                'std': np.std(confidences),
                'quartiles': np.percentile(confidences, [25, 50, 75]).tolist()
            }
            
            # Calibration metrics
            conf_bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for i in range(len(conf_bins) - 1):
                bin_indices = [j for j, conf in enumerate(confidences) 
                              if conf_bins[i] <= conf < conf_bins[i+1]]
                if bin_indices:
                    bin_accuracy = np.mean([1 if true_indices[j] == pred_indices[j] else 0 
                                           for j in bin_indices])
                    bin_avg_conf = np.mean([confidences[j] for j in bin_indices])
                    bin_accuracies.append(bin_accuracy)
                    bin_confidences.append(bin_avg_conf)
                    bin_counts.append(len(bin_indices))
                else:
                    bin_accuracies.append(0)
                    bin_confidences.append(0)
                    bin_counts.append(0)
            
            # Expected Calibration Error (ECE)
            ece = sum([(count / len(confidences)) * abs(acc - conf) 
                      for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences) 
                      if count > 0])
            
            confidence_metrics = {
                'average_confidence': avg_confidence,
                'average_confidence_correct': avg_confidence_correct,
                'average_confidence_incorrect': avg_confidence_incorrect,
                'confidence_distribution': confidence_distribution,
                'bin_accuracies': bin_accuracies,
                'bin_confidences': bin_confidences,
                'bin_counts': bin_counts,
                'expected_calibration_error': ece
            }
        
        # Combine all metrics
        metrics = {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'precision': {
                'weighted': precision_weighted,
                'macro': precision_macro,
                'per_class': {label: classification_rep[label]['precision'] for label in self.sentiment_labels}
            },
            'recall': {
                'weighted': recall_weighted,
                'macro': recall_macro,
                'per_class': {label: classification_rep[label]['recall'] for label in self.sentiment_labels}
            },
            'f1_score': {
                'weighted': f1_weighted,
                'macro': f1_macro,
                'per_class': {label: classification_rep[label]['f1-score'] for label in self.sentiment_labels}
            },
            'support': {label: int(classification_rep[label]['support']) for label in self.sentiment_labels},
            'confusion_matrix': conf_matrix.tolist(),
            'normalized_confusion_matrix': conf_matrix_norm.tolist(),
            'true_class_distribution': {label: true_class_distribution[label] 
                                       for label in self.sentiment_labels if label in true_class_distribution},
            'predicted_class_distribution': {label: pred_class_distribution[label] 
                                           for label in self.sentiment_labels if label in pred_class_distribution},
            'error_analysis': error_analysis
        }
        
        # Add confidence metrics if available
        if confidence_metrics:
            metrics['confidence'] = confidence_metrics
        
        return metrics
    
    def plot_confusion_matrix(self, confusion_matrix, title="Confusion Matrix", normalize=False, 
                             figsize=(10, 8), cmap=plt.cm.Blues, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: The confusion matrix to plot
            title: Title for the plot
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size as (width, height)
            cmap: Color map for the plot
            save_path: Path to save the figure (if None, figure is not saved)
        """
        plt.figure(figsize=figsize)
        
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Replace NaNs with zeros
            plt.title(f"Normalized {title}")
        else:
            cm = confusion_matrix
            plt.title(title)
            
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(self.sentiment_labels))
        plt.xticks(tick_marks, self.sentiment_labels, rotation=45)
        plt.yticks(tick_marks, self.sentiment_labels)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
                
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_metrics(self, metrics, title="Model Performance Metrics", 
                    figsize=(15, 10), save_path=None):
        """
        Plot comprehensive model performance metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
            title: Title for the plot
            figsize: Figure size as (width, height)
            save_path: Path to save the figure (if None, figure is not saved)
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # 1. Precision, Recall, F1 per class
        ax1 = fig.add_subplot(2, 2, 1)
        labels = list(metrics['precision']['per_class'].keys())
        precision_values = [metrics['precision']['per_class'][label] for label in labels]
        recall_values = [metrics['recall']['per_class'][label] for label in labels]
        f1_values = [metrics['f1_score']['per_class'][label] for label in labels]
        
        x = np.arange(len(labels))
        width = 0.25
        ax1.bar(x - width, precision_values, width, label='Precision')
        ax1.bar(x, recall_values, width, label='Recall')
        ax1.bar(x + width, f1_values, width, label='F1')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylim(0, 1.1)
        ax1.set_title('Per-Class Metrics')
        ax1.legend()
        
        # 2. Class distribution
        ax2 = fig.add_subplot(2, 2, 2)
        true_counts = [metrics['true_class_distribution'].get(label, 0) for label in labels]
        pred_counts = [metrics['predicted_class_distribution'].get(label, 0) for label in labels]
        
        ax2.bar(x - width/2, true_counts, width, label='True')
        ax2.bar(x + width/2, pred_counts, width, label='Predicted')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_title('Class Distribution')
        ax2.legend()
        
        # 3. Overall Metrics
        ax3 = fig.add_subplot(2, 2, 3)
        overall_metrics = [
            metrics['accuracy'], 
            metrics['precision']['weighted'], 
            metrics['recall']['weighted'], 
            metrics['f1_score']['weighted']
        ]
        metric_names = ['Accuracy', 'Precision (W)', 'Recall (W)', 'F1 (W)']
        ax3.bar(metric_names, overall_metrics)
        ax3.set_ylim(0, 1.1)
        ax3.set_title('Overall Metrics')
        
        # 4. Error Analysis
        ax4 = fig.add_subplot(2, 2, 4)
        if 'confidence' in metrics:
            conf_data = metrics['confidence']
            bins = np.linspace(0, 1, 11)[:-1]  # 10 bins from 0 to 1
            bin_accs = conf_data['bin_accuracies']
            bin_confs = conf_data['bin_confidences']
            
            width = 0.35
            ax4.bar(bins, bin_accs, width, label='Accuracy')
            ax4.bar(bins + width, bin_confs, width, label='Avg. Confidence')
            ax4.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1.1)
            ax4.set_title('Confidence Calibration')
            ax4.set_xlabel('Confidence')
            ax4.set_ylabel('Accuracy')
            ax4.legend()
        else:
            # If no confidence data, show error distribution
            error_counts = []
            error_labels = []
            for true_label, errors in metrics['error_analysis'].items():
                for pred_label, count in errors.items():
                    error_labels.append(f"{true_label}→{pred_label}")
                    error_counts.append(count)
            
            # Sort by count
            sorted_indices = np.argsort(error_counts)[::-1][:5]  # Top 5 errors
            top_labels = [error_labels[i] for i in sorted_indices]
            top_counts = [error_counts[i] for i in sorted_indices]
            
            ax4.barh(top_labels, top_counts)
            ax4.set_title('Top 5 Error Types')
            ax4.set_xlabel('Count')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_evaluation_report(self, metrics, detailed_report=True):
        """
        Generate a human-readable evaluation report
        
        Args:
            metrics: Dictionary of evaluation metrics
            detailed_report: Whether to include detailed metrics
            
        Returns:
            String with formatted evaluation report
        """
        # Basic metrics
        report = "===== MODEL EVALUATION REPORT =====\n\n"
        report += f"Overall Accuracy: {metrics['accuracy']:.4f}\n"
        report += f"Error Rate: {metrics['error_rate']:.4f}\n\n"
        
        # Weighted metrics
        report += "=== WEIGHTED METRICS ===\n"
        report += f"Precision (weighted): {metrics['precision']['weighted']:.4f}\n"
        report += f"Recall (weighted): {metrics['recall']['weighted']:.4f}\n"
        report += f"F1-Score (weighted): {metrics['f1_score']['weighted']:.4f}\n\n"
        
        # Per-class metrics
        report += "=== PER-CLASS METRICS ===\n"
        for label in self.sentiment_labels:
            report += f"--- {label} ---\n"
            report += f"Precision: {metrics['precision']['per_class'][label]:.4f}\n"
            report += f"Recall: {metrics['recall']['per_class'][label]:.4f}\n"
            report += f"F1-Score: {metrics['f1_score']['per_class'][label]:.4f}\n"
            report += f"Support: {metrics['support'][label]}\n\n"
        
        # Class distribution
        report += "=== CLASS DISTRIBUTION ===\n"
        report += "True Distribution:\n"
        for label in self.sentiment_labels:
            count = metrics['true_class_distribution'].get(label, 0)
            percentage = count / sum(metrics['true_class_distribution'].values()) * 100 if count else 0
            report += f"  {label}: {count} ({percentage:.1f}%)\n"
        
        report += "\nPredicted Distribution:\n"
        for label in self.sentiment_labels:
            count = metrics['predicted_class_distribution'].get(label, 0)
            percentage = count / sum(metrics['predicted_class_distribution'].values()) * 100 if count else 0
            report += f"  {label}: {count} ({percentage:.1f}%)\n"
        
        # Error analysis
        if metrics['error_analysis']:
            report += "\n=== ERROR ANALYSIS ===\n"
            for true_label, errors in metrics['error_analysis'].items():
                report += f"True class '{true_label}' misclassified as:\n"
                for pred_label, count in errors.items():
                    report += f"  {pred_label}: {count}\n"
                report += "\n"
        
        # Confidence metrics
        if 'confidence' in metrics:
            conf_data = metrics['confidence']
            report += "=== CONFIDENCE METRICS ===\n"
            report += f"Average Confidence: {conf_data['average_confidence']:.4f}\n"
            report += f"Average Confidence (Correct Predictions): {conf_data['average_confidence_correct']:.4f}\n"
            if conf_data['average_confidence_incorrect'] > 0:
                report += f"Average Confidence (Incorrect Predictions): {conf_data['average_confidence_incorrect']:.4f}\n"
            report += f"Expected Calibration Error: {conf_data['expected_calibration_error']:.4f}\n"
            
            report += "\nConfidence Distribution:\n"
            dist = conf_data['confidence_distribution']
            report += f"  Min: {dist['min']:.4f}\n"
            report += f"  Max: {dist['max']:.4f}\n"
            report += f"  Mean: {dist['mean']:.4f}\n"
            report += f"  Median: {dist['median']:.4f}\n"
            report += f"  Std: {dist['std']:.4f}\n"
            report += f"  Q1: {dist['quartiles'][0]:.4f}\n"
            report += f"  Q3: {dist['quartiles'][2]:.4f}\n"
        
        # Include confusion matrix for detailed report
        if detailed_report:
            report += "\n=== CONFUSION MATRIX ===\n"
            conf_matrix = np.array(metrics['confusion_matrix'])
            header = "TRUE \\ PRED | " + " | ".join(f"{label[:5]}" for label in self.sentiment_labels)
            report += header + "\n"
            report += "-" * len(header) + "\n"
            for i, label in enumerate(self.sentiment_labels):
                row = f"{label[:8]}" + " " * max(0, 8 - len(label[:8])) + " | "
                row += " | ".join(f"{conf_matrix[i, j]:5d}" for j in range(len(self.sentiment_labels)))
                report += row + "\n"
            
            report += "\n=== NORMALIZED CONFUSION MATRIX ===\n"
            conf_matrix_norm = np.array(metrics['normalized_confusion_matrix'])
            header = "TRUE \\ PRED | " + " | ".join(f"{label[:5]}" for label in self.sentiment_labels)
            report += header + "\n"
            report += "-" * len(header) + "\n"
            for i, label in enumerate(self.sentiment_labels):
                row = f"{label[:8]}" + " " * max(0, 8 - len(label[:8])) + " | "
                row += " | ".join(f"{conf_matrix_norm[i, j]:.3f}" for j in range(len(self.sentiment_labels)))
                report += row + "\n"
        
        return report
    
    def save_metrics(self, metrics, file_path):
        """
        Save metrics to a JSON file
        
        Args:
            metrics: Dictionary of evaluation metrics
            file_path: Path to save the metrics JSON file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to {file_path}")

    def evaluate_multilingual_performance(self, predictions, true_labels, languages):
        """
        Evaluate model performance across different languages
        
        Args:
            predictions: List of predicted labels
            true_labels: List of true labels
            languages: List of language codes for each prediction
            
        Returns:
            Dictionary with per-language metrics
        """
        # Get unique languages
        unique_langs = set(languages)
        
        # Evaluate performance for each language
        lang_metrics = {}
        for lang in unique_langs:
            # Get indices for this language
            lang_indices = [i for i, l in enumerate(languages) if l == lang]
            
            if len(lang_indices) > 0:
                # Extract predictions and true labels for this language
                lang_predictions = [predictions[i] for i in lang_indices]
                lang_true_labels = [true_labels[i] for i in lang_indices]
                
                # Calculate metrics
                lang_metrics[lang] = {
                    'count': len(lang_indices),
                    'accuracy': accuracy_score(lang_true_labels, lang_predictions),
                    'precision': precision_score(
                        [self.label_to_idx.get(label, 0) for label in lang_true_labels],
                        [self.label_to_idx.get(label, 0) for label in lang_predictions],
                        average='weighted', zero_division=0
                    ),
                    'recall': recall_score(
                        [self.label_to_idx.get(label, 0) for label in lang_true_labels],
                        [self.label_to_idx.get(label, 0) for label in lang_predictions],
                        average='weighted', zero_division=0
                    ),
                    'f1': f1_score(
                        [self.label_to_idx.get(label, 0) for label in lang_true_labels],
                        [self.label_to_idx.get(label, 0) for label in lang_predictions],
                        average='weighted', zero_division=0
                    )
                }
        
        return {
            'per_language': lang_metrics,
            'language_distribution': Counter(languages)
        }

    def evaluate_cross_disaster_performance(self, predictions, true_labels, disaster_types):
        """
        Evaluate model performance across different disaster types
        
        Args:
            predictions: List of predicted labels
            true_labels: List of true labels
            disaster_types: List of disaster types for each prediction
            
        Returns:
            Dictionary with per-disaster metrics
        """
        # Get unique disaster types
        unique_disasters = set(disaster_types)
        
        # Evaluate performance for each disaster type
        disaster_metrics = {}
        for disaster in unique_disasters:
            # Get indices for this disaster type
            disaster_indices = [i for i, d in enumerate(disaster_types) if d == disaster]
            
            if len(disaster_indices) > 0:
                # Extract predictions and true labels for this disaster type
                disaster_predictions = [predictions[i] for i in disaster_indices]
                disaster_true_labels = [true_labels[i] for i in disaster_indices]
                
                # Calculate metrics
                disaster_metrics[disaster] = {
                    'count': len(disaster_indices),
                    'accuracy': accuracy_score(disaster_true_labels, disaster_predictions),
                    'precision': precision_score(
                        [self.label_to_idx.get(label, 0) for label in disaster_true_labels],
                        [self.label_to_idx.get(label, 0) for label in disaster_predictions],
                        average='weighted', zero_division=0
                    ),
                    'recall': recall_score(
                        [self.label_to_idx.get(label, 0) for label in disaster_true_labels],
                        [self.label_to_idx.get(label, 0) for label in disaster_predictions],
                        average='weighted', zero_division=0
                    ),
                    'f1': f1_score(
                        [self.label_to_idx.get(label, 0) for label in disaster_true_labels],
                        [self.label_to_idx.get(label, 0) for label in disaster_predictions],
                        average='weighted', zero_division=0
                    )
                }
        
        return {
            'per_disaster': disaster_metrics,
            'disaster_distribution': Counter(disaster_types)
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--true_labels", type=str, required=True, help="CSV file with true labels")
    parser.add_argument("--predictions", type=str, required=True, help="CSV file with predictions")
    parser.add_argument("--output", type=str, default="evaluation_metrics.json", help="Output JSON file")
    parser.add_argument("--label_column", type=str, default="sentiment", help="Column name for sentiment labels")
    parser.add_argument("--confidence_column", type=str, default="confidence", help="Column name for confidence scores")
    parser.add_argument("--language_column", type=str, default="language", help="Column name for language")
    parser.add_argument("--disaster_column", type=str, default="disaster_type", help="Column name for disaster type")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--plot_dir", type=str, default="evaluation_plots", help="Directory for plot outputs")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Load true labels and predictions
        true_df = pd.read_csv(args.true_labels)
        pred_df = pd.read_csv(args.predictions)
        
        # Extract labels and confidences
        true_labels = true_df[args.label_column].tolist()
        pred_labels = pred_df[args.label_column].tolist()
        
        confidences = None
        if args.confidence_column in pred_df.columns:
            confidences = pred_df[args.confidence_column].tolist()
        
        # Evaluate
        metrics = evaluator.evaluate_predictions(true_labels, pred_labels, confidences)
        
        # Calculate multilingual performance if language column is available
        if args.language_column in true_df.columns:
            languages = true_df[args.language_column].tolist()
            multilingual_metrics = evaluator.evaluate_multilingual_performance(pred_labels, true_labels, languages)
            metrics['multilingual'] = multilingual_metrics
        
        # Calculate cross-disaster performance if disaster type column is available
        if args.disaster_column in true_df.columns:
            disaster_types = true_df[args.disaster_column].tolist()
            disaster_metrics = evaluator.evaluate_cross_disaster_performance(pred_labels, true_labels, disaster_types)
            metrics['cross_disaster'] = disaster_metrics
        
        # Save metrics
        evaluator.save_metrics(metrics, args.output)
        
        # Generate and save evaluation report
        report = evaluator.generate_evaluation_report(metrics)
        with open(args.output.replace('.json', '_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        
        # Generate plots if requested
        if args.plot:
            import os
            os.makedirs(args.plot_dir, exist_ok=True)
            
            # Confusion matrix
            evaluator.plot_confusion_matrix(
                np.array(metrics['confusion_matrix']),
                save_path=os.path.join(args.plot_dir, 'confusion_matrix.png')
            )
            
            # Normalized confusion matrix
            evaluator.plot_confusion_matrix(
                np.array(metrics['normalized_confusion_matrix']),
                normalize=True,
                save_path=os.path.join(args.plot_dir, 'normalized_confusion_matrix.png')
            )
            
            # Metrics plots
            evaluator.plot_metrics(
                metrics,
                save_path=os.path.join(args.plot_dir, 'performance_metrics.png')
            )
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)