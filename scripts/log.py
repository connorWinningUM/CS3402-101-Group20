import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_learning_curves(results_df):
    datasets = results_df['dataset'].unique()
    
    for dataset in datasets:
        subset = results_df[results_df['dataset'] == dataset]
        models = subset['model'].unique()
        
        plt.figure(figsize=(10, 6))
        
        for model in models:
            model_data = subset[subset['model'] == model]
            
            # Group by training size to get mean and std across the 3 runs
            stats = model_data.groupby("train_size").agg({
                "train_error": ["mean", "std"],
                "test_error": ["mean", "std"]
            }).sort_index()
            
            sizes = stats.index
            
            # Plot Training Error
            line, = plt.plot(sizes, stats[('train_error', 'mean')], 
                             label=f'{model} - Train', linestyle='--')
            plt.fill_between(sizes, 
                             stats[('train_error', 'mean')] - stats[('train_error', 'std')],
                             stats[('train_error', 'mean')] + stats[('train_error', 'std')], 
                             alpha=0.1, color=line.get_color())
            
            # Plot Test Error
            plt.plot(sizes, stats[('test_error', 'mean')], 
                     label=f'{model} - Test', marker='o', color=line.get_color())
            plt.fill_between(sizes, 
                             stats[('test_error', 'mean')] - stats[('test_error', 'std')],
                             stats[('test_error', 'mean')] + stats[('test_error', 'std')], 
                             alpha=0.2, color=line.get_color())

        plt.title(f"Learning Curves: {dataset}")
        plt.xlabel("Training Set Size")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.yscale('log')  # Often helpful if error ranges are large
        plt.tight_layout()
        plt.show()

def plot_model_comparison(results_df):
    max_size = results_df['train_size'].max()
    final_results = results_df[results_df['train_size'] == max_size]
    
    pivot_df = final_results.groupby(['dataset', 'model'])['test_error'].mean().unstack()
    
    pivot_df.plot(kind='bar', figsize=(12, 6))
    plt.title(f"Model Comparison at Training Size: {max_size}")
    plt.ylabel("Mean Squared Error (Test)")
    plt.xticks(rotation=45)
    plt.legend(title="Model Type")
    plt.tight_layout()
    plt.show()