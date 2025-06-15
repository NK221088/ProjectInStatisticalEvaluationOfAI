import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

def perform_kruskal_wallis_analysis_single_df(df, category_column, domain_name, feature_columns, alpha=0.05):
    """
    Perform Kruskal-Wallis tests for each feature within a domain using a single DataFrame
    with categorical group labels.
   
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with countries as rows, features as columns, and a category column
    category_column : str
        Name of the column containing group labels (e.g., 'economic_category')
    domain_name : str
        Name of the domain being analyzed (e.g., 'Economic')
    feature_columns : list
        List of feature column names to analyze
    alpha : float
        Significance level (default: 0.05)
   
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with test results including raw p-values, FDR-corrected p-values,
        H-statistics, and effect sizes
    """
   
    # Get unique groups and filter out any with value 0 (unassigned)
    unique_groups = sorted([g for g in df[category_column].unique() if g > 0])
    
    results = {
        'Feature': [],
        'H_statistic': [],
        'Raw_p_value': [],
        'FDR_p_value': [],
        'Significant_raw': [],
        'Significant_FDR': [],
        'Effect_size_eta_squared': [],
        'Group_counts': [],
        'Group_means': []
    }
   
    print(f"\n=== {domain_name} Domain Analysis ===")
    print(f"Groups found: {unique_groups}")
    print(f"Features to analyze: {len(feature_columns)}")
    
    # Print group sizes
    for group in unique_groups:
        group_size = len(df[df[category_column] == group])
        print(f"Group {group}: {group_size} countries")
   
    # Store raw p-values for FDR correction
    raw_p_values = []
   
    # Perform Kruskal-Wallis test for each feature
    for feature in feature_columns:
        if feature not in df.columns:
            print(f"Warning: {feature} not found in DataFrame")
            continue
            
        print(f"\nAnalyzing feature: {feature}")
       
        # Collect data for this feature from all groups
        groups_data = []
        group_info = []
        valid_groups = []
        
        for group in unique_groups:
            group_data = df[df[category_column] == group][feature].dropna()
            
            if len(group_data) > 0:  # Only include groups with data
                groups_data.append(group_data.values)
                group_mean = group_data.mean()
                group_median = group_data.median()
                group_info.append(f"Group {group}: n={len(group_data)}, Mean={group_mean:.3f}, Median={group_median:.3f}")
                valid_groups.append(group)
                print(f"  Group {group}: {len(group_data)} countries, "
                      f"Mean={group_mean:.3f}, Median={group_median:.3f}")
            else:
                print(f"  Group {group}: 0 countries (skipped)")
       
        # Need at least 2 groups with data for Kruskal-Wallis
        if len(groups_data) < 2:
            print(f"  Skipping {feature}: insufficient groups with data (need ≥2, have {len(groups_data)})")
            continue
           
        # Perform Kruskal-Wallis test
        try:
            h_stat, p_value = stats.kruskal(*groups_data)
           
            # Calculate effect size (eta-squared approximation for Kruskal-Wallis)
            n_total = sum(len(group) for group in groups_data)
            eta_squared = (h_stat - len(groups_data) + 1) / (n_total - len(groups_data))
            eta_squared = max(0, eta_squared)  # Ensure non-negative
           
            # Store results
            results['Feature'].append(feature)
            results['H_statistic'].append(h_stat)
            results['Raw_p_value'].append(p_value)
            results['Effect_size_eta_squared'].append(eta_squared)
            results['Significant_raw'].append(p_value < alpha)
            results['Group_counts'].append([len(group) for group in groups_data])
            results['Group_means'].append([np.mean(group) for group in groups_data])
           
            raw_p_values.append(p_value)
           
            print(f"  H-statistic: {h_stat:.4f}, p-value: {p_value:.6f}")
            
        except Exception as e:
            print(f"  Error analyzing {feature}: {str(e)}")
   
    # Apply FDR correction (Benjamini-Hochberg)
    if raw_p_values:
        rejected, fdr_p_values, alpha_sidak, alpha_bonf = multipletests(
            raw_p_values, alpha=alpha, method='fdr_bh'
        )
       
        # Add FDR results to our results dictionary
        results['FDR_p_value'] = fdr_p_values.tolist()
        results['Significant_FDR'] = rejected.tolist()
       
        print(f"\n=== FDR Correction Results ===")
        print(f"Original significant features (α={alpha}): {sum(results['Significant_raw'])}")
        print(f"FDR-corrected significant features: {sum(results['Significant_FDR'])}")
    else:
        print(f"\n=== No valid tests performed ===")
        return pd.DataFrame()  # Return empty DataFrame
   
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Raw_p_value')  # Sort by p-value
   
    return results_df

def print_results_summary(results_df, domain_name):
    """Print a formatted summary of results"""
    if results_df.empty:
        print(f"\n=== {domain_name} - No Results to Display ===")
        return
        
    print(f"\n=== {domain_name} - Final Results Summary ===")
    print(f"{'Feature':<25} {'H-stat':<8} {'Raw p':<10} {'FDR p':<10} {'η²':<8} {'Sig (FDR)'}")
    print("-" * 75)
   
    for _, row in results_df.iterrows():
        sig_marker = "***" if row['Significant_FDR'] else ""
        print(f"{row['Feature']:<25} {row['H_statistic']:<8.3f} "
              f"{row['Raw_p_value']:<10.6f} {row['FDR_p_value']:<10.6f} "
              f"{row['Effect_size_eta_squared']:<8.3f} {sig_marker}")

def analyze_domain_single_df(df, category_column, domain_name, feature_columns, alpha=0.05):
    """
    Convenience function to analyze a domain using a single DataFrame with category labels
   
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with countries as rows, features as columns, and category column
    category_column : str
        Name of column containing group labels
    domain_name : str
        Name of the domain
    feature_columns : list
        List of feature column names
    alpha : float
        Significance level
    """
   
    # Perform analysis
    results = perform_kruskal_wallis_analysis_single_df(
        df, category_column, domain_name, feature_columns, alpha
    )
   
    # Print summary
    print_results_summary(results, domain_name)
   
    return results

def create_visualization(df, results_df, category_column, domain_name, top_n=5):
    """
    Create visualizations for the most significant features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame
    results_df : pandas.DataFrame
        Results from Kruskal-Wallis analysis
    category_column : str
        Name of category column
    domain_name : str
        Domain name for plot titles
    top_n : int
        Number of top significant features to plot
    """
    
    if results_df.empty:
        print("No results to visualize")
        return
    
    # Get top significant features
    significant_features = results_df[results_df['Significant_FDR'] == True]
    
    if len(significant_features) == 0:
        print("No significant features to visualize")
        return
    
    # Take top N most significant (lowest p-values)
    top_features = significant_features.head(top_n)['Feature'].tolist()
    
    # Create subplots
    n_features = len(top_features)
    if n_features == 0:
        return
        
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    for i, feature in enumerate(top_features):
        # Create box plot
        groups_data = []
        labels = []
        
        unique_groups = sorted([g for g in df[category_column].unique() if g > 0])
        
        for group in unique_groups:
            group_data = df[df[category_column] == group][feature].dropna()
            if len(group_data) > 0:
                groups_data.append(group_data.values)
                labels.append(f'Group {group}')
        
        axes[i].boxplot(groups_data, labels=labels)
        axes[i].set_title(f'{feature} - {domain_name} Domain')
        axes[i].set_ylabel(feature)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage with your data format:
if __name__ == "__main__":
    
    # Define your feature columns (adjust based on your actual column names)
    feature_columns = [
        'sentiment_polarity',
        'flesch_reading_ease',
        '!',
        'grammatical_analysis',
        'academic', 
        'vocational',
        'userConsiderations',
        'background', 
        'international',
        'otherTypesOfSchool'
        # Add other feature columns as needed
    ]
    
    # Load your categorized data
    print("Loading categorized data...")
    df = pd.read_csv('categorized_data.csv')  # Load the CSV we created
    
    print(f"Data loaded successfully: {len(df)} countries, {len(df.columns)} features")
    print(f"Available columns: {list(df.columns)}")
    
    # Analyze Economic Domain
    print("\nAnalyzing Economic Categories...")
    economic_results = analyze_domain_single_df(
        df, 
        'economic_category', 
        'Economic', 
        feature_columns
    )
    
    # Analyze Geographical Domain  
    print("\nAnalyzing Geographical Categories...")
    geographical_results = analyze_domain_single_df(
        df, 
        'geographical_category', 
        'Geographical', 
        feature_columns
    )
    
    # Analyze Educational Domain
    print("\nAnalyzing Educational Categories...")
    educational_results = analyze_domain_single_df(
        df, 
        'educational_category', 
        'Educational', 
        feature_columns
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualization(df, economic_results, 'economic_category', 'Economic')
    create_visualization(df, geographical_results, 'geographical_category', 'Geographical')  
    create_visualization(df, educational_results, 'educational_category', 'Educational')
    
    # Save results
    print("\nSaving results...")
    economic_results.to_csv('economic_category_results.csv', index=False)
    geographical_results.to_csv('geographical_category_results.csv', index=False)
    educational_results.to_csv('educational_category_results.csv', index=False)
    
    print("\nAnalysis complete! Results saved to CSV files.")
    print("Files created:")
    print("- economic_category_results.csv")
    print("- geographical_category_results.csv") 
    print("- educational_category_results.csv")