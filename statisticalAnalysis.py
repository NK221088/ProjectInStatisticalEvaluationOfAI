import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations

# Try to import scikit-posthocs for Dunn's test
try:
    import scikit_posthocs as sp
    POSTHOC_AVAILABLE = True
except ImportError:
    POSTHOC_AVAILABLE = False
    print("Warning: scikit-posthocs not available. Install with: pip install scikit-posthocs")
    print("Post-hoc analysis will use Mann-Whitney U tests as fallback.")

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

def perform_posthoc_analysis(df, category_column, feature, alpha=0.05, method='dunn'):
    """
    Perform post-hoc analysis after significant Kruskal-Wallis test
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data
    category_column : str
        Name of column containing group labels
    feature : str
        Feature column name to analyze
    alpha : float
        Significance level
    method : str
        'dunn' for Dunn's test, 'mannwhitney' for pairwise Mann-Whitney U
    
    Returns:
    --------
    results : dict or DataFrame
        Post-hoc test results
    """
    
    # Get unique groups (excluding 0)
    unique_groups = sorted([g for g in df[category_column].unique() if g > 0])
    
    if method == 'dunn' and POSTHOC_AVAILABLE:
        # Prepare data for Dunn's test
        # scikit-posthocs expects a list of arrays
        groups_data = []
        group_labels = []
        
        for group in unique_groups:
            group_data = df[df[category_column] == group][feature].dropna()
            if len(group_data) > 0:
                groups_data.extend(group_data.values)
                group_labels.extend([f'Group_{group}'] * len(group_data))
        
        # Create DataFrame for scikit-posthocs
        posthoc_df = pd.DataFrame({
            'values': groups_data,
            'groups': group_labels
        })
        
        # Perform Dunn's test
        dunn_results = sp.posthoc_dunn(posthoc_df, val_col='values', group_col='groups', 
                                      p_adjust='fdr_bh')  # FDR correction
        
        return dunn_results
    
    else:
        # Perform pairwise Mann-Whitney U tests (fallback or explicit choice)
        results = {
            'Group_1': [],
            'Group_2': [],
            'U_statistic': [],
            'p_value': [],
            'effect_size': []
        }
        
        # Get all pairwise combinations
        for group1, group2 in combinations(unique_groups, 2):
            data1 = df[df[category_column] == group1][feature].dropna()
            data2 = df[df[category_column] == group2][feature].dropna()
            
            if len(data1) > 0 and len(data2) > 0:
                # Perform Mann-Whitney U test
                u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                # Calculate effect size (rank-biserial correlation)
                n1, n2 = len(data1), len(data2)
                effect_size = 1 - (2 * u_stat) / (n1 * n2)
                
                results['Group_1'].append(group1)
                results['Group_2'].append(group2)
                results['U_statistic'].append(u_stat)
                results['p_value'].append(p_val)
                results['effect_size'].append(effect_size)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Apply multiple comparison correction
        if len(results_df) > 0:
            rejected, corrected_p, _, _ = multipletests(
                results_df['p_value'].values, alpha=alpha, method='fdr_bh'
            )
            results_df['p_value_corrected'] = corrected_p
            results_df['significant'] = rejected
        
        return results_df

def analyze_significant_features_posthoc(df, results_df, category_column, domain_name, alpha=0.05):
    """
    Perform post-hoc analysis for all significant features from Kruskal-Wallis results
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original data
    results_df : pandas.DataFrame
        Results from Kruskal-Wallis analysis
    category_column : str
        Category column name
    domain_name : str
        Domain name for printing
    alpha : float
        Significance level
    """
    
    # Get significant features
    significant_features = results_df[results_df['Significant_FDR'] == True]
    
    if len(significant_features) == 0:
        print(f"No significant features found for {domain_name} domain")
        return {}
    
    print(f"\n=== Post-Hoc Analysis for {domain_name} Domain ===")
    print(f"Analyzing {len(significant_features)} significant features")
    print("NOTE: Each feature is analyzed separately with its own multiple comparison correction")
    
    posthoc_results = {}
    
    for idx, (_, row) in enumerate(significant_features.iterrows(), 1):
        feature = row['Feature']
        print(f"\n--- [{idx}/{len(significant_features)}] Post-hoc analysis for {feature} ---")
        print(f"Original Kruskal-Wallis: H = {row['H_statistic']:.3f}, p = {row['FDR_p_value']:.6f}")
        
        # Determine which method to use
        method = 'dunn' if POSTHOC_AVAILABLE else 'mannwhitney'
        
        # Perform post-hoc analysis
        try:
            posthoc_result = perform_posthoc_analysis(df, category_column, feature, 
                                                    alpha=alpha, method=method)
            posthoc_results[f"{feature}_{method}"] = posthoc_result
            
            if method == 'dunn':
                print("Dunn's test results (FDR-corrected p-values):")
                print(posthoc_result.round(4))
                
                # Identify significant pairs
                significant_pairs = []
                for i in range(len(posthoc_result.index)):
                    for j in range(len(posthoc_result.columns)):
                        if i < j:  # Only upper triangle to avoid duplicates
                            p_val = posthoc_result.iloc[i, j]
                            if p_val < alpha:
                                group1 = posthoc_result.index[i]
                                group2 = posthoc_result.columns[j]
                                significant_pairs.append((group1, group2, p_val))
                
                if significant_pairs:
                    print(f"\nSignificant pairwise differences (p < {alpha}):")
                    for group1, group2, p_val in significant_pairs:
                        print(f"  {group1} vs {group2}: p = {p_val:.4f}")
                else:
                    print(f"No significant pairwise differences found (p < {alpha})")
            
            else:  # Mann-Whitney U method
                print("Mann-Whitney U pairwise comparisons:")
                significant_mw = posthoc_result[posthoc_result['significant'] == True]
                if len(significant_mw) > 0:
                    for _, mw_row in significant_mw.iterrows():
                        print(f"  Group {mw_row['Group_1']} vs Group {mw_row['Group_2']}: "
                              f"p = {mw_row['p_value_corrected']:.4f}, "
                              f"effect size = {mw_row['effect_size']:.3f}")
                else:
                    print("  No significant pairwise differences found")
                
        except Exception as e:
            print(f"Error in post-hoc analysis for {feature}: {str(e)}")
    
    return posthoc_results

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

def analyze_domain_single_df(df, category_column, domain_name, feature_columns, alpha=0.05, 
                           perform_posthoc=True):
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
    perform_posthoc : bool
        Whether to perform post-hoc analysis for significant features
        
    Returns:
    --------
    results : pandas.DataFrame
        Kruskal-Wallis results
    posthoc_results : dict
        Post-hoc analysis results (if perform_posthoc=True)
    """
   
    # Perform Kruskal-Wallis analysis
    results = perform_kruskal_wallis_analysis_single_df(
        df, category_column, domain_name, feature_columns, alpha
    )
   
    # Print summary
    print_results_summary(results, domain_name)
    
    # Perform post-hoc analysis if requested and there are significant results
    posthoc_results = {}
    if perform_posthoc and not results.empty:
        posthoc_results = analyze_significant_features_posthoc(
            df, results, category_column, domain_name, alpha
        )
   
    return results, posthoc_results

def create_visualization(df, results_df, category_column, domain_name, top_n=5, save_plots=True, output_dir="plots"):
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
    save_plots : bool
        Whether to save plots as PDFs
    output_dir : str
        Directory to save plots
    """
    
    if results_df.empty:
        print("No results to visualize")
        return
    
    # Create output directory if it doesn't exist
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
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
    
    # Save plot as PDF if requested
    if save_plots:
        filename = f"{domain_name.lower()}_domain_analysis.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Plot saved: {filepath}")
    
    plt.show()

def save_results(results_df, posthoc_results, domain_name, output_dir="results"):
    """
    Save both Kruskal-Wallis and post-hoc results to files
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Kruskal-Wallis results
    posthoc_results : dict
        Post-hoc analysis results
    domain_name : str
        Domain name for file naming
    output_dir : str
        Directory to save results
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save Kruskal-Wallis results
    kw_filename = f"{domain_name.lower()}_kruskal_wallis_results.csv"
    kw_filepath = os.path.join(output_dir, kw_filename)
    results_df.to_csv(kw_filepath, index=False)
    print(f"Kruskal-Wallis results saved: {kw_filepath}")
    
    # Save post-hoc results
    if posthoc_results:
        for feature_method, result in posthoc_results.items():
            if isinstance(result, pd.DataFrame):
                # Mann-Whitney U results
                filename = f"{domain_name.lower()}_{feature_method}_posthoc.csv"
                filepath = os.path.join(output_dir, filename)
                result.to_csv(filepath, index=False)
                print(f"Post-hoc results saved: {filepath}")
            else:
                # Dunn's test results (matrix)
                filename = f"{domain_name.lower()}_{feature_method}_posthoc.csv"
                filepath = os.path.join(output_dir, filename)
                result.to_csv(filepath)
                print(f"Post-hoc results saved: {filepath}")

def generate_overall_summary(df, feature_columns):
    """
    Generate overall dataset summary statistics for Section 2.1
    """
    print("="*50)
    print("OVERALL DATASET SUMMARY STATISTICS")
    print("="*50)
    
    # Basic dataset info
    print(f"Total number of countries: {len(df)}")
    print(f"Total number of features: {len(feature_columns)}")
    
    # Overall statistics for continuous features
    summary_stats = df[feature_columns].describe()
    
    # More detailed statistics for key features
    print("\nKey Feature Statistics:")
    print("-" * 30)
    
    for feature in feature_columns:
        stats = df[feature].describe()
        print(f"{feature}:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std:  {stats['std']:.3f}")
        print(f"  Min:  {stats['min']:.3f}")
        print(f"  Max:  {stats['max']:.3f}")
        print()
    
    return summary_stats

def generate_group_summary_table(df, feature_columns, category_columns):
    """
    Generate comprehensive summary statistics by groups for Section 2.2
    Returns formatted tables showing mean, std, min, max, and other statistics across all groupings
    """
    print("="*50)
    print("GROUP-SPECIFIC SUMMARY STATISTICS")
    print("="*50)
    
    # Create mapping dictionaries for better labels
    economic_mapping = {1: 'Low-income', 2: 'Lower-middle', 3: 'Upper-middle', 4: 'High-income'}
    geographical_mapping = {
        1: 'South Asia', 2: 'Europe/CA/NA', 3: 'MENA', 
        4: 'Sub-Saharan Africa', 5: 'LAC', 6: 'East Asia/Pacific'
    }
    educational_mapping = {1: 'Low (<50%)', 2: 'Medium (50-90%)', 3: 'High (>90%)'}
    
    # Store results for table creation
    results = []
    summary_tables = {}
    
    # Process each grouping domain
    for category_col, mapping, domain_name in [
        ('economic_category', economic_mapping, 'Economic'),
        ('geographical_category', geographical_mapping, 'Geographical'), 
        ('educational_category', educational_mapping, 'Educational')
    ]:
        print(f"\n{domain_name} Domain:")
        print("-" * 40)
        
        # Calculate comprehensive group statistics
        group_stats = df.groupby(category_col)[feature_columns].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ])
        
        # Display with proper labels
        for group_id, group_name in mapping.items():
            if group_id in group_stats.index:
                print(f"\n{group_name} (n={group_stats.loc[group_id, (feature_columns[0], 'count')]:.0f}):")
                
                for feature in feature_columns:
                    count = group_stats.loc[group_id, (feature, 'count')]
                    mean_val = group_stats.loc[group_id, (feature, 'mean')]
                    std_val = group_stats.loc[group_id, (feature, 'std')]
                    min_val = group_stats.loc[group_id, (feature, 'min')]
                    max_val = group_stats.loc[group_id, (feature, 'max')]
                    median_val = group_stats.loc[group_id, (feature, 'median')]
                    
                    print(f"  {feature}:")
                    print(f"    Mean: {mean_val:.3f} (SD: {std_val:.3f})")
                    print(f"    Range: {min_val:.3f} - {max_val:.3f}, Median: {median_val:.3f}")
                    
                    # Store for comprehensive table
                    results.append({
                        'Domain': domain_name,
                        'Group': group_name,
                        'Feature': feature,
                        'N': count,
                        'Mean': mean_val,
                        'Std': std_val,
                        'Min': min_val,
                        'Max': max_val,
                        'Median': median_val,
                        'CV': std_val/mean_val if mean_val != 0 else np.nan  # Coefficient of variation
                    })
        
        # Create domain-specific summary table
        domain_summary = pd.DataFrame()
        for stat in ['mean', 'std']:
            stat_data = group_stats.xs(stat, level=1, axis=1)
            stat_data.index = stat_data.index.map(mapping)
            stat_data.columns = [f"{col}_{stat}" for col in stat_data.columns]
            domain_summary = pd.concat([domain_summary, stat_data], axis=1)
        
        summary_tables[domain_name] = domain_summary
    
    # Create comprehensive results dataframe
    results_df = pd.DataFrame(results)
    
    # Create mean comparison table (for easy paper inclusion)
    mean_table = results_df.pivot_table(
        index='Feature', 
        columns=['Domain', 'Group'], 
        values='Mean', 
        aggfunc='first'
    )
    
    # Create std comparison table
    std_table = results_df.pivot_table(
        index='Feature', 
        columns=['Domain', 'Group'], 
        values='Std', 
        aggfunc='first'
    )
    
    return mean_table, std_table, summary_tables, results_df

def save_summary_statistics(overall_stats, mean_table, std_table, summary_tables, results_df, output_dir):
    """
    Save comprehensive summary statistics to CSV files
    """
    # Save overall statistics
    overall_stats.to_csv(f"{output_dir}/overall_summary_statistics.csv")
    
    # Save group summary tables
    mean_table.to_csv(f"{output_dir}/group_means_comparison.csv")
    std_table.to_csv(f"{output_dir}/group_std_comparison.csv")
    
    # Save domain-specific tables
    for domain_name, table in summary_tables.items():
        table.to_csv(f"{output_dir}/{domain_name.lower()}_domain_summary.csv")
    
    # Save comprehensive detailed results
    results_df.to_csv(f"{output_dir}/comprehensive_group_statistics.csv", index=False)
    
    # Create a publication-ready summary table combining mean and std
    publication_table = pd.DataFrame()
    for domain in ['Economic', 'Geographical', 'Educational']:
        domain_data = results_df[results_df['Domain'] == domain]
        for feature in domain_data['Feature'].unique():
            feature_data = domain_data[domain_data['Feature'] == feature]
            row_data = {}
            for _, row in feature_data.iterrows():
                col_name = f"{row['Group']}"
                row_data[col_name] = f"{row['Mean']:.3f} ({row['Std']:.3f})"
            publication_table = pd.concat([
                publication_table, 
                pd.DataFrame([row_data], index=[f"{domain}_{feature}"])
            ])
    
    publication_table.to_csv(f"{output_dir}/publication_ready_summary.csv")
    
    print(f"\nComprehensive summary statistics saved to {output_dir}/")
    print("Files created:")
    print("- overall_summary_statistics.csv")
    print("- group_means_comparison.csv")
    print("- group_std_comparison.csv") 
    print("- comprehensive_group_statistics.csv")
    print("- publication_ready_summary.csv (Mean (SD) format)")
    print("- Domain-specific CSV files for each grouping")

def create_summary_comparison_plots(df, feature_columns, output_dir):
    """
    Create comparison plots for the most important features across groups
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Select key features for visualization (you can modify this)
    key_features = ['academic', 'vocational', 'sentiment_subjectivity', 'flesch_reading_ease']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Key Feature Distributions Across Country Groups', fontsize=16, fontweight='bold')
    
    # Economic groups
    ax1 = axes[0, 0]
    economic_data = []
    economic_labels = []
    for i in range(1, 5):
        subset = df[df['economic_category'] == i][key_features[0]]
        if len(subset) > 0:
            economic_data.append(subset)
            economic_labels.append(['Low-income', 'Lower-middle', 'Upper-middle', 'High-income'][i-1])
    
    ax1.boxplot(economic_data, labels=economic_labels)
    ax1.set_title(f'{key_features[0]} by Economic Group')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add more plots for other domains and features...
    # (You can expand this based on your specific needs)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_statistics_plots.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Summary plots saved to {output_dir}/summary_statistics_plots.pdf")

# Example usage with your data format:
if __name__ == "__main__":
    
    # Define your feature columns (adjust based on your actual column names)
    feature_columns = [
        'sentiment_subjectivity',
        'flesch_reading_ease',
        'grammatical_analysis',
        'academic', 
        'vocational',
        'userConsiderations',
        'bridging',
        'international',
        'otherTypesOfSchool',
    ]

    # Columns: ['country', 'sentiment_subjectivity', 'flesch_reading_ease', 'grammatical_analysis', 'academic', 'vocational', 'userConsiderations', 'bridging', 'international', 'otherTypesOfSchool', 'economic_category', 'geographical_category', 'educational_category']
    
    # Create output directories
    plots_dir = "plots"
    results_dir = "results"
    
    for directory in [plots_dir, results_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Load your categorized data
    print("Loading categorized data...")
    df = pd.read_csv('categorized_data.csv')  # Load the CSV we created
    
    print(f"Data loaded successfully: {len(df)} countries, {len(df.columns)} features")
    print(f"Available columns: {list(df.columns)}")
    
    # Generate overall summary statistics (for Section 2.1)
    print("\n" + "="*60)
    print("GENERATING SUMMARY STATISTICS")
    print("="*60)
    
    overall_stats = generate_overall_summary(df, feature_columns)
    
    # Generate comprehensive group-specific summary statistics (for Section 2.2)
    mean_table, std_table, summary_tables, detailed_results = generate_group_summary_table(
        df, feature_columns, 
        ['economic_category', 'geographical_category', 'educational_category']
    )
    
    # Save comprehensive summary statistics
    save_summary_statistics(overall_stats, mean_table, std_table, summary_tables, detailed_results, results_dir)
    
    # Create summary comparison plots
    create_summary_comparison_plots(df, feature_columns, plots_dir)
    
    print("\nComprehensive summary statistics generated successfully!")
    print(f"Check {results_dir}/ for CSV files and {plots_dir}/ for plots")

    # Analyze Economic Domain with Post-Hoc
    print("\n" + "="*60)
    print("ANALYZING ECONOMIC CATEGORIES")
    print("="*60)
    economic_results, economic_posthoc = analyze_domain_single_df(
        df, 
        'economic_category', 
        'Economic', 
        feature_columns,
        perform_posthoc=True
    )
    
    # Analyze Geographical Domain with Post-Hoc
    print("\n" + "="*60)
    print("ANALYZING GEOGRAPHICAL CATEGORIES")
    print("="*60)
    geographical_results, geographical_posthoc = analyze_domain_single_df(
        df, 
        'geographical_category', 
        'Geographical', 
        feature_columns,
        perform_posthoc=True
    )
    
    # Analyze Educational Domain with Post-Hoc
    print("\n" + "="*60)
    print("ANALYZING EDUCATIONAL CATEGORIES")
    print("="*60)
    educational_results, educational_posthoc = analyze_domain_single_df(
        df, 
        'educational_category', 
        'Educational', 
        feature_columns,
        perform_posthoc=True
    )
    
    # Create visualizations and save as PDFs
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    create_visualization(df, economic_results, 'economic_category', 'Economic', 
                        save_plots=True, output_dir=plots_dir)
    create_visualization(df, geographical_results, 'geographical_category', 'Geographical',
                        save_plots=True, output_dir=plots_dir)  
    create_visualization(df, educational_results, 'educational_category', 'Educational',
                        save_plots=True, output_dir=plots_dir)
    
    # Save all results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    save_results(economic_results, economic_posthoc, 'Economic', results_dir)
    save_results(geographical_results, geographical_posthoc, 'Geographical', results_dir)
    save_results(educational_results, educational_posthoc, 'Educational', results_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print("\nPlots (PDF format):")
    print(f"- {plots_dir}/economic_domain_analysis.pdf")
    print(f"- {plots_dir}/geographical_domain_analysis.pdf")
    print(f"- {plots_dir}/educational_domain_analysis.pdf")
    print("\nKruskal-Wallis Results (CSV format):")
    print(f"- {results_dir}/economic_kruskal_wallis_results.csv")
    print(f"- {results_dir}/geographical_kruskal_wallis_results.csv") 
    print(f"- {results_dir}/educational_kruskal_wallis_results.csv")
    print("\nPost-Hoc Results (CSV format):")
    print("- Individual files for each significant feature in each domain")
    
    if not POSTHOC_AVAILABLE:
        print("\nNOTE: Install scikit-posthocs for Dunn's test:")
        print("pip install scikit-posthocs")