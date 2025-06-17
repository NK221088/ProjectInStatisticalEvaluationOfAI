import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Read the data
df = pd.read_csv('categorized_data.csv')

# Define the mappings
economic_mapping = {
    1: 'Low-income',
    2: 'Lower-middle-income', 
    3: 'Upper-middle-income',
    4: 'High-income'
}

geographical_mapping = {
    1: 'South Asia',
    2: 'Europe, Central Asia and North America',
    3: 'Middle East and North Africa', 
    4: 'Sub-Saharan Africa',
    5: 'Latin America and Caribbean',
    6: 'East Asia and Pacific'
}

educational_mapping = {
    1: 'Low level (<50%)',
    2: 'Medium level (50-90%)',
    3: 'High level (>90%)'
}

# Apply mappings
df['Economic_Group'] = df['economic_category'].map(economic_mapping)
df['Geographical_Group'] = df['geographical_category'].map(geographical_mapping)
df['Educational_Group'] = df['educational_category'].map(educational_mapping)

# Create a comprehensive table showing country distributions
print("=== COUNTRY DISTRIBUTION ANALYSIS ===\n")

# 1. Cross-tabulation: Economic vs Geographical
print("1. Economic vs Geographical Distribution:")
econ_geo_crosstab = pd.crosstab(df['Economic_Group'], df['Geographical_Group'], margins=True)
print(econ_geo_crosstab)
print("\n")

# 2. Cross-tabulation: Economic vs Educational
print("2. Economic vs Educational Distribution:")
econ_edu_crosstab = pd.crosstab(df['Economic_Group'], df['Educational_Group'], margins=True)
print(econ_edu_crosstab)
print("\n")

# 3. Cross-tabulation: Geographical vs Educational
print("3. Geographical vs Educational Distribution:")
geo_edu_crosstab = pd.crosstab(df['Geographical_Group'], df['Educational_Group'], margins=True)
print(geo_edu_crosstab)
print("\n")

# 4. Three-way distribution (sample countries for each combination)
print("4. Three-way Distribution (with sample countries):")
three_way = df.groupby(['Economic_Group', 'Geographical_Group', 'Educational_Group']).agg({
    'country': ['count', lambda x: ', '.join(x.head(3).tolist())]
}).round(2)
three_way.columns = ['Count', 'Sample Countries']
print(three_way)
print("\n")

# Generate LaTeX tables
print("=== LATEX TABLES ===\n")

# LaTeX table for Economic vs Geographical
print("LaTeX Table 1: Economic vs Geographical Distribution")
print("\\begin{table}[H]")
print("\\centering")
print("\\caption{Cross-tabulation: Economic vs Geographical Distribution}")
print("\\begin{tabular}{l" + "c" * len(geographical_mapping) + "c}")
print("\\toprule")
print("\\textbf{Economic Group} & " + " & ".join([f"\\textbf{{{v}}}" for v in geographical_mapping.values()]) + " & \\textbf{Total} \\\\")
print("\\midrule")

for econ_group in economic_mapping.values():
    row = [econ_group]
    for geo_group in geographical_mapping.values():
        count = len(df[(df['Economic_Group'] == econ_group) & (df['Geographical_Group'] == geo_group)])
        row.append(str(count))
    total = len(df[df['Economic_Group'] == econ_group])
    row.append(str(total))
    print(" & ".join(row) + " \\\\")

print("\\midrule")
# Total row
total_row = ["Total"]
for geo_group in geographical_mapping.values():
    count = len(df[df['Geographical_Group'] == geo_group])
    total_row.append(str(count))
total_row.append(str(len(df)))
print(" & ".join(total_row) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:EconomicGeographical}")
print("\\end{table}")
print("\n")

# LaTeX table for Economic vs Educational
print("LaTeX Table 2: Economic vs Educational Distribution")
print("\\begin{table}[H]")
print("\\centering")
print("\\caption{Cross-tabulation: Economic vs Educational Distribution}")
print("\\begin{tabular}{l" + "c" * len(educational_mapping) + "c}")
print("\\toprule")
print("\\textbf{Economic Group} & " + " & ".join([f"\\textbf{{{v}}}" for v in educational_mapping.values()]) + " & \\textbf{Total} \\\\")
print("\\midrule")

for econ_group in economic_mapping.values():
    row = [econ_group]
    for edu_group in educational_mapping.values():
        count = len(df[(df['Economic_Group'] == econ_group) & (df['Educational_Group'] == edu_group)])
        row.append(str(count))
    total = len(df[df['Economic_Group'] == econ_group])
    row.append(str(total))
    print(" & ".join(row) + " \\\\")

print("\\midrule")
# Total row
total_row = ["Total"]
for edu_group in educational_mapping.values():
    count = len(df[df['Educational_Group'] == edu_group])
    total_row.append(str(count))
total_row.append(str(len(df)))
print(" & ".join(total_row) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:EconomicEducational}")
print("\\end{table}")
print("\n")

# Create improved visualizations
fig = plt.figure(figsize=(20, 16))

# Create a 3x2 grid for better organization
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.2)

# Title for the entire figure
fig.suptitle('Country Distribution Analysis Across Different Groupings', fontsize=20, fontweight='bold', y=0.95)

# 1. Economic vs Geographical heatmap
ax1 = fig.add_subplot(gs[0, 0])
econ_geo_matrix = pd.crosstab(df['Economic_Group'], df['Geographical_Group'])
sns.heatmap(econ_geo_matrix, annot=True, fmt='d', cmap='YlOrRd', 
            ax=ax1, cbar_kws={'label': 'Number of Countries'})
ax1.set_title('Economic vs Geographical Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Geographical Group')
ax1.set_ylabel('Economic Group')

# 2. Economic vs Educational heatmap
ax2 = fig.add_subplot(gs[0, 1])
econ_edu_matrix = pd.crosstab(df['Economic_Group'], df['Educational_Group'])
sns.heatmap(econ_edu_matrix, annot=True, fmt='d', cmap='YlGnBu',
            ax=ax2, cbar_kws={'label': 'Number of Countries'})
ax2.set_title('Economic vs Educational Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Educational Group')
ax2.set_ylabel('Economic Group')

# 3. Geographical vs Educational heatmap
ax3 = fig.add_subplot(gs[1, :])
geo_edu_matrix = pd.crosstab(df['Geographical_Group'], df['Educational_Group'])
sns.heatmap(geo_edu_matrix, annot=True, fmt='d', cmap='viridis',
            ax=ax3, cbar_kws={'label': 'Number of Countries'})
ax3.set_title('Geographical vs Educational Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Educational Group')
ax3.set_ylabel('Geographical Group')

# 4. Individual distribution bar charts (improved version)
# Economic distribution
ax4 = fig.add_subplot(gs[2, 0])
econ_counts = df['Economic_Group'].value_counts()
bars1 = ax4.bar(range(len(econ_counts)), econ_counts.values, color='skyblue', alpha=0.8, edgecolor='navy')
ax4.set_title('Economic Groups Distribution', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Countries')
ax4.set_xticks(range(len(econ_counts)))
ax4.set_xticklabels([label.replace('-', '-\n') for label in econ_counts.index], rotation=0, ha='center', fontsize=10)

# Add value labels on bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Educational distribution
ax5 = fig.add_subplot(gs[2, 1])
edu_counts = df['Educational_Group'].value_counts()
bars2 = ax5.bar(range(len(edu_counts)), edu_counts.values, color='lightgreen', alpha=0.8, edgecolor='darkgreen')
ax5.set_title('Educational Groups Distribution', fontsize=12, fontweight='bold')
ax5.set_ylabel('Number of Countries')
ax5.set_xticks(range(len(edu_counts)))
ax5.set_xticklabels([label.replace(' ', '\n') for label in edu_counts.index], rotation=0, ha='center', fontsize=10)

# Add value labels on bars
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.show()

# Create a separate plot for geographical distribution (since it has more categories)
fig2, ax = plt.subplots(figsize=(12, 6))
geo_counts = df['Geographical_Group'].value_counts()
bars = ax.bar(range(len(geo_counts)), geo_counts.values, color='lightcoral', alpha=0.8, edgecolor='darkred')
ax.set_title('Geographical Groups Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Countries')
ax.set_xlabel('Geographical Group')
ax.set_xticks(range(len(geo_counts)))
ax.set_xticklabels([label.replace(' and ', '\n&\n').replace(', ', ',\n') for label in geo_counts.index], 
                   rotation=45, ha='right', fontsize=10)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Print some interesting insights
print("=== KEY INSIGHTS ===")
print(f"1. Total countries analyzed: {len(df)}")
print(f"2. Most common economic group: {df['Economic_Group'].mode().iloc[0]} ({df['Economic_Group'].value_counts().iloc[0]} countries)")
print(f"3. Most common geographical group: {df['Geographical_Group'].mode().iloc[0]} ({df['Geographical_Group'].value_counts().iloc[0]} countries)")
print(f"4. Most common educational group: {df['Educational_Group'].mode().iloc[0]} ({df['Educational_Group'].value_counts().iloc[0]} countries)")

# Find some interesting patterns
print("\n=== INTERESTING PATTERNS ===")
print("High-income countries by region:")
high_income = df[df['Economic_Group'] == 'High-income']
print(high_income['Geographical_Group'].value_counts())

print("\nLow-income countries by region:")
low_income = df[df['Economic_Group'] == 'Low-income']
print(low_income['Geographical_Group'].value_counts())

print("\nEducational levels in Sub-Saharan Africa:")
ssa_countries = df[df['Geographical_Group'] == 'Sub-Saharan Africa']
print(ssa_countries['Educational_Group'].value_counts())

print("\nEducational levels in High-income countries:")
high_income_edu = df[df['Economic_Group'] == 'High-income']
print(high_income_edu['Educational_Group'].value_counts())