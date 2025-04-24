"""
Evaluation Metrics for PanicSense Hybrid Neural Network Model
This module provides comprehensive evaluation metrics for assessing model performance
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

class ModelEvaluator:
    """Class for evaluating PanicSense model performance with comprehensive metrics"""
    
    def __init__(self, sentiment_labels=None):
        """
        Initialize the evaluator
        
        Args:
            sentiment_labels: List of sentiment label categories
        """
        self.sentiment_labels = sentiment_labels or ['Panic', 'Fear/Anxiety', 'Resilience', 'Neutral', 'Disbelief']
        self.results_dir = 'evaluation_results'
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
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
        if not true_labels or not predicted_labels:
            return {"error": "Empty input data"}
            
        if len(true_labels) != len(predicted_labels):
            return {"error": f"Length mismatch: {len(true_labels)} true labels vs {len(predicted_labels)} predictions"}
            
        # Ensure confidences exist, use 1.0 if not provided
        if confidences is None:
            confidences = [1.0] * len(true_labels)
        
        # Calculate basic metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Calculate precision, recall, and F1 score
        precision = precision_score(true_labels, predicted_labels, 
                                    labels=self.sentiment_labels, average=None, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, 
                              labels=self.sentiment_labels, average=None, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, 
                      labels=self.sentiment_labels, average=None, zero_division=0)
        
        # Calculate weighted averages
        weighted_precision = precision_score(true_labels, predicted_labels, 
                                           average='weighted', zero_division=0)
        weighted_recall = recall_score(true_labels, predicted_labels, 
                                      average='weighted', zero_division=0)
        weighted_f1 = f1_score(true_labels, predicted_labels, 
                              average='weighted', zero_division=0)
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=self.sentiment_labels)
        
        # Calculate support for each class
        support = {}
        for label in self.sentiment_labels:
            support[label] = true_labels.count(label)
            
        # Calculate confidence metrics
        avg_confidence = sum(confidences) / len(confidences)
        confidence_distribution = {
            "0.0-0.2": len([c for c in confidences if c <= 0.2]),
            "0.2-0.4": len([c for c in confidences if 0.2 < c <= 0.4]),
            "0.4-0.6": len([c for c in confidences if 0.4 < c <= 0.6]),
            "0.6-0.8": len([c for c in confidences if 0.6 < c <= 0.8]),
            "0.8-1.0": len([c for c in confidences if c > 0.8])
        }
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, label in enumerate(self.sentiment_labels):
            per_class_metrics[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": support[label]
            }
            
        # Language-specific metrics (placeholder - would need language labels)
        language_metrics = {"overall": weighted_f1}
        
        # Compile all metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision": {
                "per_class": {label: float(p) for label, p in zip(self.sentiment_labels, precision)},
                "weighted": float(weighted_precision)
            },
            "recall": {
                "per_class": {label: float(r) for label, r in zip(self.sentiment_labels, recall)},
                "weighted": float(weighted_recall)
            },
            "f1_score": {
                "per_class": {label: float(f) for label, f in zip(self.sentiment_labels, f1)},
                "weighted": float(weighted_f1)
            },
            "support": support,
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "labels": self.sentiment_labels
            },
            "confidence": {
                "average": float(avg_confidence),
                "distribution": confidence_distribution
            },
            "language_performance": language_metrics,
            "per_class_metrics": per_class_metrics
        }
        
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
        
        # Normalize if requested
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
            
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=14)
        plt.colorbar()
        
        tick_marks = np.arange(len(self.sentiment_labels))
        plt.xticks(tick_marks, self.sentiment_labels, rotation=45, fontsize=10)
        plt.yticks(tick_marks, self.sentiment_labels, fontsize=10)
        
        # Add labels to each cell
        thresh = confusion_matrix.max() / 2
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], fmt),
                         ha="center", va="center", 
                         color="white" if confusion_matrix[i, j] > thresh else "black",
                         fontsize=10)
                
        plt.tight_layout()
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Plot per-class precision, recall, and F1-score
        class_metrics = {
            'Precision': [metrics['precision']['per_class'][label] for label in self.sentiment_labels],
            'Recall': [metrics['recall']['per_class'][label] for label in self.sentiment_labels],
            'F1-Score': [metrics['f1_score']['per_class'][label] for label in self.sentiment_labels]
        }
        
        x = np.arange(len(self.sentiment_labels))
        width = 0.25
        
        axs[0, 0].bar(x - width, class_metrics['Precision'], width, label='Precision')
        axs[0, 0].bar(x, class_metrics['Recall'], width, label='Recall')
        axs[0, 0].bar(x + width, class_metrics['F1-Score'], width, label='F1-Score')
        
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(self.sentiment_labels, rotation=45, ha='right')
        axs[0, 0].set_ylim(0, 1.0)
        axs[0, 0].set_title('Per-Class Performance')
        axs[0, 0].legend()
        
        # Plot class distribution
        support = [metrics['support'][label] for label in self.sentiment_labels]
        axs[0, 1].bar(x, support, width*2)
        axs[0, 1].set_xticks(x)
        axs[0, 1].set_xticklabels(self.sentiment_labels, rotation=45, ha='right')
        axs[0, 1].set_title('Class Distribution (Support)')
        
        # Plot confusion matrix
        conf_matrix = np.array(metrics['confusion_matrix']['matrix'])
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        im = axs[1, 0].imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues)
        axs[1, 0].set_title('Normalized Confusion Matrix')
        
        tick_marks = np.arange(len(self.sentiment_labels))
        axs[1, 0].set_xticks(tick_marks)
        axs[1, 0].set_xticklabels(self.sentiment_labels, rotation=45, ha='right')
        axs[1, 0].set_yticks(tick_marks)
        axs[1, 0].set_yticklabels(self.sentiment_labels)
        
        fmt = '.2f'
        thresh = conf_matrix_norm.max() / 2
        for i in range(conf_matrix_norm.shape[0]):
            for j in range(conf_matrix_norm.shape[1]):
                axs[1, 0].text(j, i, format(conf_matrix_norm[i, j], fmt),
                         ha="center", va="center", 
                         color="white" if conf_matrix_norm[i, j] > thresh else "black")
        
        # Plot confidence distribution
        confidence_dist = metrics['confidence']['distribution']
        axs[1, 1].bar(confidence_dist.keys(), confidence_dist.values())
        axs[1, 1].set_title(f'Confidence Distribution (Avg: {metrics["confidence"]["average"]:.2f})')
        axs[1, 1].set_xlabel('Confidence Range')
        axs[1, 1].set_ylabel('Count')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
        if 'error' in metrics:
            return f"Error in evaluation: {metrics['error']}"
            
        report = []
        
        # Add report header
        report.append("=" * 50)
        report.append("     PanicSense Hybrid Model Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Add overall metrics
        report.append("Overall Performance:")
        report.append(f"- Accuracy: {metrics['accuracy']:.4f}")
        report.append(f"- Weighted Precision: {metrics['precision']['weighted']:.4f}")
        report.append(f"- Weighted Recall: {metrics['recall']['weighted']:.4f}")
        report.append(f"- Weighted F1-Score: {metrics['f1_score']['weighted']:.4f}")
        report.append(f"- Average Confidence: {metrics['confidence']['average']:.4f}")
        report.append("")
        
        # Add per-class metrics if detailed report is requested
        if detailed_report:
            report.append("Per-Class Performance:")
            report.append("-" * 50)
            report.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            report.append("-" * 50)
            
            for label in self.sentiment_labels:
                precision = metrics['precision']['per_class'][label]
                recall = metrics['recall']['per_class'][label]
                f1 = metrics['f1_score']['per_class'][label]
                support = metrics['support'][label]
                
                report.append(f"{label:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
            
            report.append("-" * 50)
            report.append("")
            
            # Add confidence distribution
            report.append("Confidence Distribution:")
            for range_name, count in metrics['confidence']['distribution'].items():
                report.append(f"- {range_name}: {count} predictions")
            report.append("")
            
        return "\n".join(report)
    
    def save_metrics(self, metrics, file_path):
        """
        Save metrics to a JSON file
        
        Args:
            metrics: Dictionary of evaluation metrics
            file_path: Path to save the metrics JSON file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
            
        print(f"Metrics saved to {file_path}")
    
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
        # Group predictions by language
        language_groups = {}
        for pred, true, lang in zip(predictions, true_labels, languages):
            if lang not in language_groups:
                language_groups[lang] = {'predictions': [], 'true_labels': []}
            
            language_groups[lang]['predictions'].append(pred)
            language_groups[lang]['true_labels'].append(true)
        
        # Calculate metrics for each language
        language_metrics = {}
        for lang, data in language_groups.items():
            if len(data['predictions']) > 10:  # Only evaluate if enough samples
                accuracy = accuracy_score(data['true_labels'], data['predictions'])
                weighted_f1 = f1_score(data['true_labels'], data['predictions'], average='weighted', zero_division=0)
                
                language_metrics[lang] = {
                    'accuracy': float(accuracy),
                    'f1_score': float(weighted_f1),
                    'sample_count': len(data['predictions'])
                }
        
        return language_metrics
    
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
        # Group predictions by disaster type
        disaster_groups = {}
        for pred, true, disaster in zip(predictions, true_labels, disaster_types):
            if disaster not in disaster_groups:
                disaster_groups[disaster] = {'predictions': [], 'true_labels': []}
            
            disaster_groups[disaster]['predictions'].append(pred)
            disaster_groups[disaster]['true_labels'].append(true)
        
        # Calculate metrics for each disaster type
        disaster_metrics = {}
        for disaster, data in disaster_groups.items():
            if len(data['predictions']) > 10:  # Only evaluate if enough samples
                accuracy = accuracy_score(data['true_labels'], data['predictions'])
                weighted_f1 = f1_score(data['true_labels'], data['predictions'], average='weighted', zero_division=0)
                
                disaster_metrics[disaster] = {
                    'accuracy': float(accuracy),
                    'f1_score': float(weighted_f1),
                    'sample_count': len(data['predictions'])
                }
        
        return disaster_metrics