import pandas
import numpy as np
# from scipy import stats
# from statsmodel.stats.multitest import multitests
import matplotlib.pyplot as plt
import seaborn as sns

def perform_kruskal_wallis_analysis(data_dict, domain_name, feature_columns, alpha=0.05):
    """
    Perform Kruskal-Wallis tests for each feature within a domain, with FDR correction.
   
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are group names (e.g., 'Group_1', 'Group_2')
        and values are DataFrames with countries as rows and features as columns
    domain_name : str
        Name of the domain being analyzed (e.g., 'Educational_Attainment')
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
   
    results = {
        'Feature': [],
        'H_statistic': [],
        'Raw_p_value': [],
        'FDR_p_value': [],
        'Significant_raw': [],
        'Significant_FDR': [],
        'Effect_size_eta_squared': []
    }
   
    print(f"\n=== {domain_name} Domain Analysis ===")
    print(f"Groups: {list(data_dict.keys())}")
    print(f"Features to analyze: {len(feature_columns)}")
   
    # Store raw p-values for FDR correction
    raw_p_values = []
   
    # Perform Kruskal-Wallis test for each feature
    for feature in feature_columns:
        print(f"\nAnalyzing feature: {feature}")
       
        # Collect data for this feature from all groups
        feature_data = []
        group_labels = []
       
        for group_name, df in data_dict.items():
            if feature in df.columns:
                feature_values = df[feature].dropna()  # Remove NaN values
                feature_data.extend(feature_values)
                group_labels.extend([group_name] * len(feature_values))
                print(f"  {group_name}: {len(feature_values)} countries, "
                      f"Mean={feature_values.mean():.3f}, Median={feature_values.median():.3f}")
            else:
                print(f"  Warning: {feature} not found in {group_name}")
       
        if len(set(group_labels)) < 2:
            print(f"  Skipping {feature}: insufficient groups")
            continue
           
        # Prepare data for Kruskal-Wallis test
        groups_data = []
        for group_name in data_dict.keys():
            if group_name in group_labels:
                group_feature_data = []
                for i, label in enumerate(group_labels):
                    if label == group_name:
                        group_feature_data.append(feature_data[i])
                if group_feature_data:  # Only add if group has data
                    groups_data.append(group_feature_data)
       
        # Perform Kruskal-Wallis test
        if len(groups_data) >= 2:
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
           
            raw_p_values.append(p_value)
           
            print(f"  H-statistic: {h_stat:.4f}, p-value: {p_value:.6f}")
        else:
            print(f"  Skipping {feature}: insufficient data")
   
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
   
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Raw_p_value')  # Sort by p-value
   
    return results_df

def print_results_summary(results_df, domain_name):
    """Print a formatted summary of results"""
    print(f"\n=== {domain_name} - Final Results Summary ===")
    print(f"{'Feature':<25} {'H-stat':<8} {'Raw p':<10} {'FDR p':<10} {'η²':<8} {'Sig (FDR)'}")
    print("-" * 75)
   
    for _, row in results_df.iterrows():
        sig_marker = "***" if row['Significant_FDR'] else ""
        print(f"{row['Feature']:<25} {row['H_statistic']:<8.3f} "
              f"{row['Raw_p_value']:<10.6f} {row['FDR_p_value']:<10.6f} "
              f"{row['Effect_size_eta_squared']:<8.3f} {sig_marker}")

def analyze_domain(group1_df, group2_df, group3_df, domain_name, feature_columns, group4_df=None):
    """
    Convenience function to analyze a domain with 3 or 4 groups
   
    Parameters:
    -----------
    group1_df, group2_df, group3_df : pandas.DataFrame
        DataFrames for each group (countries as rows, features as columns)
    group4_df : pandas.DataFrame, optional
        Fourth group (for domains like GNI with 4 groups)
    domain_name : str
        Name of the domain
    feature_columns : list
        List of feature column names
    """
   
    # Create data dictionary
    data_dict = {
        'Group_1': group1_df,
        'Group_2': group2_df,
        'Group_3': group3_df
    }
   
    if group4_df is not None:
        data_dict['Group_4'] = group4_df
   
    # Perform analysis
    results = perform_kruskal_wallis_analysis(data_dict, domain_name, feature_columns)
   
    # Print summary
    print_results_summary(results, domain_name)
   
    return results

# Example usage:
if __name__ == "__main__":
    # Define your features (adjust based on your actual column names)
    feature_columns = [
        'unique_word_count',
        'token_count',
        'sentiment_polarity',
        'flesch_reading_ease',
        'academic_references',
        'vocational_references',
        'international_perspectives',
        'background_interest'
    ]
   
    # Example: Educational Attainment Domain Analysis
    # Replace these with your actual DataFrames
    """
    edu_low_df = pd.read_csv('educational_low_group.csv')  # Countries with <50% education
    edu_medium_df = pd.read_csv('educational_medium_group.csv')  # Countries with 50-90% education  
    edu_high_df = pd.read_csv('educational_high_group.csv')  # Countries with >90% education
   
    edu_results = analyze_domain(
        edu_low_df, edu_medium_df, edu_high_df,
        "Educational_Attainment",
        feature_columns
    )
   
    # Example: GNI Domain Analysis (4 groups)
    gni_group1_df = pd.read_csv('gni_group1.csv')
    gni_group2_df = pd.read_csv('gni_group2.csv')
    gni_group3_df = pd.read_csv('gni_group3.csv')
    gni_group4_df = pd.read_csv('gni_group4.csv')
   
    gni_results = analyze_domain(
        gni_group1_df, gni_group2_df, gni_group3_df,
        "GNI",
        feature_columns,
        group4_df=gni_group4_df
    )
   
    # Save results
    edu_results.to_csv('educational_attainment_results.csv', index=False)
    gni_results.to_csv('gni_results.csv', index=False)
    """