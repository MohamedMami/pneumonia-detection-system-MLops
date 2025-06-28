import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.evaluation_results = None
    
    def evaluate_model(self, test_generator, save_results=True):
        """Comprehensive model evaluation"""
        
        print("Starting model evaluation...")
        
        # Reset generator
        test_generator.reset()
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        
        # Get true labels
        true_labels = test_generator.classes
        
        # Convert predictions to binary
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        prediction_scores = predictions.flatten()
        
        # Calculate metrics
        results = self._calculate_metrics(true_labels, predicted_labels, prediction_scores)
        
        # Add additional info
        results['total_samples'] = len(true_labels)
        results['class_distribution'] = {
            'Normal': int(np.sum(true_labels == 0)),
            'Pneumonia': int(np.sum(true_labels == 1))
        }
        
        self.evaluation_results = results
        
        if save_results:
            self._save_evaluation_results(results)
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, average_precision_score)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'average_precision': float(average_precision_score(y_true, y_pred_proba))
        }
        
        # Sensitivity and Specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['sensitivity'] = float(tp / (tp + fn))  # Same as recall
        metrics['specificity'] = float(tn / (tn + fp))
        metrics['false_positive_rate'] = float(fp / (fp + tn))
        metrics['false_negative_rate'] = float(fn / (fn + tp))
        
        # Positive and Negative Predictive Values
        metrics['positive_predictive_value'] = float(tp / (tp + fp))  # Same as precision
        metrics['negative_predictive_value'] = float(tn / (tn + fn))
        
        return metrics
    
    def _save_evaluation_results(self, results):
        """Save evaluation results to file"""
        
        evaluation_data = {
            'evaluation_date': datetime.now().isoformat(),
            'model_config': {
                'img_height': self.config.IMG_HEIGHT,
                'img_width': self.config.IMG_WIDTH,
                'batch_size': self.config.BATCH_SIZE
            },
            'metrics': results
        }
        
        results_path = os.path.join(self.config.RESULTS_DIR, 'evaluation_results.json')
        
        with open(results_path, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        print(f"Evaluation results saved to {results_path}")
    
    def print_evaluation_summary(self):
        """Print a formatted evaluation summary"""
        
        if self.evaluation_results is None:
            print("No evaluation results available. Run evaluate_model() first.")
            return
        
        results = self.evaluation_results
        
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Total Samples: {results['total_samples']}")
        print(f"Normal Cases: {results['class_distribution']['Normal']}")
        print(f"Pneumonia Cases: {results['class_distribution']['Pneumonia']}")
        
        print("\nPERFORMANCE METRICS:")
        print("-" * 30)
        print(f"Accuracy:      {results['accuracy']:.4f}")
        print(f"Precision:     {results['precision']:.4f}")
        print(f"Recall:        {results['recall']:.4f}")
        print(f"F1-Score:      {results['f1_score']:.4f}")
        print(f"ROC AUC:       {results['roc_auc']:.4f}")
        print(f"Avg Precision: {results['average_precision']:.4f}")
        
        print("\nCLINICAL METRICS:")
        print("-" * 30)
        print(f"Sensitivity:   {results['sensitivity']:.4f}")
        print(f"Specificity:   {results['specificity']:.4f}")
        print(f"PPV:           {results['positive_predictive_value']:.4f}")
        print(f"NPV:           {results['negative_predictive_value']:.4f}")
        
        print("\nERROR RATES:")
        print("-" * 30)
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        print(f"False Negative Rate: {results['false_negative_rate']:.4f}")
        
        print("="*60)
    
    def generate_detailed_report(self, test_generator, save_path=None):
        """Generate detailed evaluation report with visualizations"""
        
        from utils.visualization import VisualizationUtils
        
        if self.evaluation_results is None:
            print("Running evaluation first...")
            self.evaluate_model(test_generator)
        
        # Initialize visualization utils
        viz = VisualizationUtils(self.config)
        
        # Get predictions for visualization
        test_generator.reset()
        predictions = self.model.predict(test_generator, verbose=0)
        true_labels = test_generator.classes
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        
        # Generate visualizations
        report_dir = os.path.join(self.config.RESULTS_DIR, 'detailed_report')
        os.makedirs(report_dir, exist_ok=True)
        
        # Confusion Matrix
        cm_path = os.path.join(report_dir, 'confusion_matrix.png')
        viz.plot_confusion_matrix(true_labels, predicted_labels, save_path=cm_path)
        
        # ROC Curve
        roc_path = os.path.join(report_dir, 'roc_curve.png')
        viz.plot_roc_curve(true_labels, predictions, save_path=roc_path)
        
        # Sample predictions (if images available)
        # This would require modifying the generator to return images
        
        print(f"Detailed report generated in {report_dir}")
        
        return report_dir
    
    def compare_models(self, other_results, model_names=None):
        """Compare evaluation results with other models"""
        
        if self.evaluation_results is None:
            print("No evaluation results available for current model.")
            return
        
        if model_names is None:
            model_names = ['Current Model'] + [f'Model {i+1}' for i in range(len(other_results))]
        
        # Prepare comparison data
        all_results = [self.evaluation_results] + other_results
        
        # Create comparison DataFrame
        import pandas as pd
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        comparison_data = []
        for i, result in enumerate(all_results):
            row = {'Model': model_names[i]}
            for metric in metrics_to_compare:
                row[metric.replace('_', ' ').title()] = result[metric]
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        print("\nMODEL COMPARISON:")
        print("="*60)
        print(df.to_string(index=False, float_format='%.4f'))
        
        return df