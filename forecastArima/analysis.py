import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

warnings.filterwarnings("ignore")

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    # Remove zeros to avoid division by zero
    mask = actual != 0
    if not mask.any():
        return np.inf
    actual = actual[mask]
    predicted = predicted[mask]
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_rmse(actual, predicted):
    """Calculate Root Mean Square Error"""
    return math.sqrt(mean_squared_error(actual, predicted))

def analyze_model_accuracy(model_path, test_data, model_name):
    """Analyze accuracy of a single model"""
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Make predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(start=len(test_data)-len(test_data)//4, end=len(test_data)-1)
        else:
            # For ARIMA models, we need to handle differently
            predictions = model.forecast(steps=len(test_data)//4)
        
        # Get actual values for comparison
        actual = test_data.iloc[-len(predictions):]
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predictions)
        mape = calculate_mape(actual, predictions)
        rmse = calculate_rmse(actual, predictions)
        
        return {
            'model_name': model_name,
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'predictions': predictions,
            'actual': actual
        }
    except Exception as e:
        print(f"Error analyzing {model_name}: {str(e)}")
        return None

def load_test_data():
    """Load test data for evaluation"""
    try:
        # Load the original data
        train = pd.read_csv("rossmann-store-sales/train.csv", parse_dates=True, low_memory=False, index_col='Date')
        
        # Filter data as in the original script
        train = train[(train["Open"] != 0) & (train['Sales'] != 0)]
        
        # Get sales data for each store
        sales_a = train[train.Store == 2]['Sales']
        sales_b = train[train.Store == 85]['Sales']
        sales_c = train[train.Store == 1]['Sales']
        sales_d = train[train.Store == 13]['Sales']
        
        return {
            'Store_A': sales_a,
            'Store_B': sales_b,
            'Store_C': sales_c,
            'Store_D': sales_d
        }
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return None

def create_accuracy_report(results):
    """Create a comprehensive accuracy report"""
    # Create results DataFrame
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to analyze")
        return
    
    df_results = pd.DataFrame(valid_results)
    
    # Save detailed results
    os.makedirs("output/accuracy_analysis", exist_ok=True)
    df_results.to_csv("output/accuracy_analysis/model_accuracy_results.csv", index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL ACCURACY ANALYSIS REPORT")
    print("="*80)
    
    for _, row in df_results.iterrows():
        print(f"\n{row['model_name']}:")
        print(f"  MAE:  {row['mae']:.2f}")
        print(f"  MAPE: {row['mape']:.2f}%")
        print(f"  RMSE: {row['rmse']:.2f}")
    
    # Find best models for each metric
    print("\n" + "="*80)
    print("BEST MODELS BY METRIC")
    print("="*80)
    
    best_mae = df_results.loc[df_results['mae'].idxmin()]
    best_mape = df_results.loc[df_results['mape'].idxmin()]
    best_rmse = df_results.loc[df_results['rmse'].idxmin()]
    
    print(f"\nBest MAE:  {best_mae['model_name']} ({best_mae['mae']:.2f})")
    print(f"Best MAPE: {best_mape['model_name']} ({best_mape['mape']:.2f}%)")
    print(f"Best RMSE: {best_rmse['model_name']} ({best_rmse['rmse']:.2f})")
    
    return df_results

def create_visualizations(results):
    """Create visualizations for model accuracy"""
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to visualize")
        return
    
    df_results = pd.DataFrame(valid_results)
    
    # Create output directory
    os.makedirs("output/accuracy_analysis", exist_ok=True)
    
    # 1. Metrics comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # MAE comparison
    axes[0].bar(df_results['model_name'], df_results['mae'])
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_ylabel('MAE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # MAPE comparison
    axes[1].bar(df_results['model_name'], df_results['mape'])
    axes[1].set_title('Mean Absolute Percentage Error (MAPE)')
    axes[1].set_ylabel('MAPE (%)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # RMSE comparison
    axes[2].bar(df_results['model_name'], df_results['rmse'])
    axes[2].set_title('Root Mean Square Error (RMSE)')
    axes[2].set_ylabel('RMSE')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/accuracy_analysis/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Actual vs Predicted plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(valid_results[:4]):  # Plot first 4 models
        if i < len(axes):
            actual = result['actual']
            predicted = result['predictions']
            
            axes[i].plot(actual.index, actual, label='Actual', color='blue', alpha=0.7)
            axes[i].plot(actual.index, predicted, label='Predicted', color='red', alpha=0.7)
            axes[i].set_title(f"{result['model_name']}\nMAE: {result['mae']:.2f}, MAPE: {result['mape']:.2f}%")
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Sales')
            axes[i].legend()
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/accuracy_analysis/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of metrics
    metrics_df = df_results[['model_name', 'mae', 'mape', 'rmse']].set_index('model_name')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df.T, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Error Value'})
    plt.title('Model Accuracy Metrics Heatmap')
    plt.tight_layout()
    plt.savefig('output/accuracy_analysis/metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to output/accuracy_analysis/")

def main():
    """Main function to run the accuracy analysis"""
    print("Starting Model Accuracy Analysis...")
    
    # Load test data
    test_data = load_test_data()
    if test_data is None:
        print("Failed to load test data. Exiting.")
        return
    
    # Define models to analyze
    models_to_analyze = [
        ('models/sarima_model_a_train.pkl', 'Store_A', 'SARIMA Store A (Train)'),
        ('models/sarima_model_b_train.pkl', 'Store_B', 'SARIMA Store B (Train)'),
        ('models/sarima_model_c_train.pkl', 'Store_C', 'SARIMA Store C (Train)'),
        ('models/sarima_model_d_train.pkl', 'Store_D', 'SARIMA Store D (Train)'),
        ('models/arima_model_a.pkl', 'Store_A', 'ARIMA Store A'),
        ('models/arima_model_b.pkl', 'Store_B', 'ARIMA Store B'),
        ('models/arima_model_c.pkl', 'Store_C', 'ARIMA Store C'),
        ('models/arima_model_d.pkl', 'Store_D', 'ARIMA Store D'),
        ('models/sarima_model_a.pkl', 'Store_A', 'SARIMA Store A'),
        ('models/sarima_model_b.pkl', 'Store_B', 'SARIMA Store B'),
        ('models/sarima_model_c.pkl', 'Store_C', 'SARIMA Store C'),
        ('models/sarima_model_d.pkl', 'Store_D', 'SARIMA Store D'),
    ]
    
    results = []
    
    # Analyze each model
    for model_path, store_key, model_name in models_to_analyze:
        if os.path.exists(model_path):
            print(f"Analyzing {model_name}...")
            result = analyze_model_accuracy(model_path, test_data[store_key], model_name)
            results.append(result)
        else:
            print(f"Model file not found: {model_path}")
    
    # Create accuracy report
    df_results = create_accuracy_report(results)
    
    # Create visualizations
    if df_results is not None:
        create_visualizations(results)
    
    print("\nModel accuracy analysis completed!")

if __name__ == "__main__":
    main() 