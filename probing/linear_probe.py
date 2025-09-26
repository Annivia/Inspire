#!/usr/bin/env python3
"""
linear_probe.py

Shared linear probing functionality for all experiments.
Provides unified interface for strictly linear regression and classification probes with
comprehensive evaluation metrics and baseline comparisons.

Design Principle: Keep probes as simple as possible (linear transformations only) 
to ensure interpretability and test what information is linearly accessible in representations.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Union, Optional
import json
import time
from pathlib import Path


class LinearProbe:
    """
    Strictly linear probe for both regression and classification tasks.
    Uses only linear transformations to test what information is linearly accessible 
    in the learned representations.
    """
    
    def __init__(self, task_type: str = 'regression', random_seed: int = 42, debug: bool = False):
        """
        Initialize linear probe.
        
        Args:
            task_type: 'regression' or 'classification'
            random_seed: Random seed for reproducibility
            debug: Enable debug output
        """
        self.task_type = task_type
        self.random_seed = random_seed
        self.debug = debug
        
        if task_type == 'regression':
            self.model = LinearRegression()
        elif task_type == 'classification':
            self.model = LogisticRegression(random_state=random_seed, max_iter=1000)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression' or 'classification'")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        if self.debug:
            print(f"[DEBUG] Initialized STRICTLY LINEAR {task_type} probe with random seed {random_seed}")
    
    def fit_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        condition_name: str = "normal"
    ) -> Dict:
        """
        Fit strictly linear probe and evaluate on held-out test set.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples, n_targets)
            test_size: Fraction of data to use for testing
            condition_name: Name of experimental condition (for logging)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.debug:
            print(f"[DEBUG] Fitting LINEAR {condition_name} probe: X={X.shape}, y={y.shape}")
        
        start_time = time.time()
        
        # Flatten features to 2D if needed: [N, ...] -> [N, D]
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed
        )
        
        if self.debug:
            print(f"[DEBUG] Train set: X={X_train.shape}, y={y_train.shape}")
            print(f"[DEBUG] Test set: X={X_test.shape}, y={y_test.shape}")
        
        # Scale features (linear preprocessing only)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.debug:
            print(f"[DEBUG] Feature scaling applied - mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
        
        # Fit linear model (NO hidden layers, NO non-linearities)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Make predictions using linear transformation only
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Compute metrics
        fit_time = time.time() - start_time
        
        if self.task_type == 'regression':
            metrics = self._compute_regression_metrics(
                y_train, y_pred_train, y_test, y_pred_test
            )
        else:
            metrics = self._compute_classification_metrics(
                y_train, y_pred_train, y_test, y_pred_test
            )
        
        # Add metadata
        metrics.update({
            'condition': condition_name,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': X.shape[1],
            'n_targets': y.shape[1] if len(y.shape) > 1 else 1,
            'fit_time': fit_time,
            'task_type': self.task_type,
            'model_type': 'strictly_linear'
        })
        
        if self.debug:
            print(f"[DEBUG] LINEAR {condition_name} probe fitted in {fit_time:.3f}s")
            if self.task_type == 'regression':
                print(f"[DEBUG] R2 score: {metrics['r2_test']:.4f} (linear separability measure)")
                print(f"[DEBUG] MSE: {metrics['mse_test']:.4f}")
            else:
                print(f"[DEBUG] Accuracy: {metrics['accuracy_test']:.4f} (linear separability measure)")
                print(f"[DEBUG] F1 score: {metrics['f1_test']:.4f}")
        
        return metrics
    
    def _compute_regression_metrics(
        self,
        y_train: np.ndarray,
        y_pred_train: np.ndarray, 
        y_test: np.ndarray,
        y_pred_test: np.ndarray
    ) -> Dict:
        """Compute regression evaluation metrics."""
        
        # Handle multi-output regression (e.g., 7-dimensional actions)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # Multi-output case - compute metrics per output and overall
            metrics = {}
            
            # Per-output metrics (e.g., for each action dimension)
            for i in range(y_test.shape[1]):
                r2_train = r2_score(y_train[:, i], y_pred_train[:, i])
                r2_test = r2_score(y_test[:, i], y_pred_test[:, i])
                mse_train = mean_squared_error(y_train[:, i], y_pred_train[:, i])
                mse_test = mean_squared_error(y_test[:, i], y_pred_test[:, i])
                
                metrics[f'r2_train_dim_{i}'] = r2_train
                metrics[f'r2_test_dim_{i}'] = r2_test
                metrics[f'mse_train_dim_{i}'] = mse_train
                metrics[f'mse_test_dim_{i}'] = mse_test
            
            # Overall metrics (average across outputs)
            metrics['r2_train'] = np.mean([metrics[f'r2_train_dim_{i}'] for i in range(y_test.shape[1])])
            metrics['r2_test'] = np.mean([metrics[f'r2_test_dim_{i}'] for i in range(y_test.shape[1])])
            metrics['mse_train'] = np.mean([metrics[f'mse_train_dim_{i}'] for i in range(y_test.shape[1])])
            metrics['mse_test'] = np.mean([metrics[f'mse_test_dim_{i}'] for i in range(y_test.shape[1])])
            
        else:
            # Single-output case
            metrics = {
                'r2_train': r2_score(y_train, y_pred_train),
                'r2_test': r2_score(y_test, y_pred_test),
                'mse_train': mean_squared_error(y_train, y_pred_train),
                'mse_test': mean_squared_error(y_test, y_pred_test)
            }
        
        return metrics
    
    def _compute_classification_metrics(
        self,
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_test: np.ndarray, 
        y_pred_test: np.ndarray
    ) -> Dict:
        """Compute classification evaluation metrics."""
        
        # Handle multi-class/multi-label case
        average_method = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
        
        metrics = {
            'accuracy_train': accuracy_score(y_train, y_pred_train),
            'accuracy_test': accuracy_score(y_test, y_pred_test),
            'f1_train': f1_score(y_train, y_pred_train, average=average_method),
            'f1_test': f1_score(y_test, y_pred_test, average=average_method),
            'precision_train': precision_score(y_train, y_pred_train, average=average_method),
            'precision_test': precision_score(y_test, y_pred_test, average=average_method),
            'recall_train': recall_score(y_train, y_pred_train, average=average_method),
            'recall_test': recall_score(y_test, y_pred_test, average=average_method)
        }
        
        return metrics


def create_baseline_data(
    X: np.ndarray, 
    y: np.ndarray, 
    baseline_type: str,
    random_seed: int = 42,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create baseline data for comparison experiments.
    
    Args:
        X: Input features
        y: Target values
        baseline_type: 'normal', 'randomized', or 'noise'
        random_seed: Random seed for reproducibility
        debug: Enable debug output
        
    Returns:
        Tuple of (X_baseline, y_baseline)
    """
    np.random.seed(random_seed)
    
    if baseline_type == 'normal':
        # Original data - test if representations linearly encode the information
        return X.copy(), y.copy()
    
    elif baseline_type == 'randomized':
        # Randomly shuffle pairs on trajectory basis - breaks correspondence
        # This tests if probe performance comes from actual correlations vs random chance
        X_shuffled = X.copy()
        y_shuffled = y.copy()
        
        # Shuffle indices to break X-y correspondence
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        X_shuffled = X_shuffled[indices]
        # Keep y in original order to break correspondence
        
        if debug:
            print(f"[DEBUG] Created randomized baseline: shuffled {len(indices)} samples")
            print(f"[DEBUG] This breaks X-y correspondence to test random chance performance")
        
        return X_shuffled, y_shuffled
    
    elif baseline_type == 'noise':
        # Replace targets with Gaussian noise - tests probe's ability to fit noise
        # Good representations should NOT be able to linearly predict noise
        if len(y.shape) == 1:
            noise_shape = y.shape
        else:
            noise_shape = y.shape
        
        y_noise = np.random.normal(0, 1, size=noise_shape)
        
        # Scale noise to have similar magnitude as original targets
        if y.std() > 0:
            y_noise = y_noise * y.std() + y.mean()
        
        if debug:
            print(f"[DEBUG] Created noise baseline: shape={noise_shape}")
            print(f"[DEBUG] Noise stats - mean={y_noise.mean():.3f}, std={y_noise.std():.3f}")
            print(f"[DEBUG] Original stats - mean={y.mean():.3f}, std={y.std():.3f}")
        
        return X.copy(), y_noise
    
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")


def run_probe_with_baselines(
    X: np.ndarray,
    y: np.ndarray,
    probe_name: str = "probe",
    task_type: str = 'regression',
    test_size: float = 0.2,
    random_seed: int = 42,
    debug: bool = False
) -> Dict:
    """
    Run strictly linear probe with all three baseline conditions.
    
    Args:
        X: Input features (e.g., hidden states, vision features)
        y: Target values (e.g., actions, visual concepts)
        probe_name: Name for this probe (for logging)
        task_type: 'regression' or 'classification'
        test_size: Fraction of data for testing
        random_seed: Random seed
        debug: Enable debug output
        
    Returns:
        Dictionary containing results for all conditions
    """
    if debug:
        print(f"[DEBUG] Running STRICTLY LINEAR {probe_name} with all baselines")
        print(f"[DEBUG] Input shape: X={X.shape}, y={y.shape}")
        print(f"[DEBUG] Testing what information is LINEARLY ACCESSIBLE in representations")
    
    results = {}
    baselines = ['normal', 'randomized', 'noise']
    
    for baseline in baselines:
        if debug:
            print(f"[DEBUG] Running {baseline} condition...")
        
        # Create baseline data
        X_baseline, y_baseline = create_baseline_data(X, y, baseline, random_seed, debug)
        
        # Create and fit strictly linear probe
        probe = LinearProbe(task_type=task_type, random_seed=random_seed, debug=debug)
        
        try:
            metrics = probe.fit_and_evaluate(
                X_baseline, y_baseline, test_size=test_size, condition_name=baseline
            )
            results[baseline] = metrics
            
            if debug:
                if task_type == 'regression':
                    print(f"[DEBUG] {baseline} R2: {metrics['r2_test']:.4f} (higher = more linearly separable)")
                else:
                    print(f"[DEBUG] {baseline} accuracy: {metrics['accuracy_test']:.4f} (higher = more linearly separable)")
                    
        except Exception as e:
            if debug:
                print(f"[DEBUG] ERROR in {baseline} condition: {e}")
            results[baseline] = {'error': str(e)}
    
    # Add interpretability-focused summary statistics
    if all(baseline in results and 'error' not in results[baseline] for baseline in baselines):
        if task_type == 'regression':
            normal_r2 = results['normal'].get('r2_test', 0)
            random_r2 = results['randomized'].get('r2_test', 0) 
            noise_r2 = results['noise'].get('r2_test', 0)
            
            results['summary'] = {
                'probe_name': probe_name,
                'task_type': task_type,
                'normal_vs_random_diff': normal_r2 - random_r2,
                'normal_vs_noise_diff': normal_r2 - noise_r2,
                'best_r2': normal_r2,
                'linear_separability_strength': normal_r2,
                'random_chance_performance': random_r2,
                'noise_overfitting': noise_r2
            }
        else:
            normal_acc = results['normal'].get('accuracy_test', 0)
            random_acc = results['randomized'].get('accuracy_test', 0)
            noise_acc = results['noise'].get('accuracy_test', 0)
            
            results['summary'] = {
                'probe_name': probe_name,
                'task_type': task_type,
                'normal_vs_random_diff': normal_acc - random_acc,
                'normal_vs_noise_diff': normal_acc - noise_acc,
                'best_accuracy': normal_acc,
                'linear_separability_strength': normal_acc,
                'random_chance_performance': random_acc,
                'noise_overfitting': noise_acc
            }
        
        if debug:
            print(f"[DEBUG] Linear separability analysis:")
            if task_type == 'regression':
                print(f"[DEBUG] - Normal R2: {normal_r2:.4f} (information linearly accessible)")
                print(f"[DEBUG] - Random R2: {random_r2:.4f} (chance performance)")
                print(f"[DEBUG] - Noise R2: {noise_r2:.4f} (overfitting to noise)")
            else:
                print(f"[DEBUG] - Normal Acc: {normal_acc:.4f} (linear separability)")
                print(f"[DEBUG] - Random Acc: {random_acc:.4f} (chance performance)")
                print(f"[DEBUG] - Noise Acc: {noise_acc:.4f} (overfitting to noise)")
    
    return results


def save_probe_results(results: Dict, output_file: Union[str, Path]):
    """Save probe results to JSON file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"[DEBUG] Saved probe results to: {output_file}")
