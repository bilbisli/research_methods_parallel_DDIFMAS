import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import numpy as np

def generate_graphs(data):
    if data.lower().endswith('.csv'):
        df = pd.read_csv(data)
    elif data.lower().endswith('.xlsx') or data.lower().endswith('.xls'):
        df = pd.read_excel(data)
    else:
        print("Error: Invalid file format. Please provide a CSV or Excel file.")
        return

    # Create a folder to save the graphs
    if not os.path.exists("runtime_comparison_graphs"):
        os.makedirs("runtime_comparison_graphs")

    # Runtime Comparison by Number of Faulty Components (nof)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='nof', y='Runtime', hue='diagnosis algorithm')
    plt.title('Runtime Comparison by Number of Faulty Components')
    plt.xlabel('Number of Faulty Components')
    plt.ylabel('Runtime (seconds)')
    plt.legend(title='Diagnosis Algorithm')
    plt.savefig('runtime_comparison_graphs/runtime_by_nof.png')
    plt.close()

    # Runtime Comparison by Agent Fault Probability (afp)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='afp', y='Runtime', hue='diagnosis algorithm')
    plt.title('Runtime Comparison by Agent Fault Probability')
    plt.xlabel('Agent Fault Probability')
    plt.ylabel('Runtime (seconds)')
    plt.legend(title='Diagnosis Algorithm')
    plt.savefig('runtime_comparison_graphs/runtime_by_afp.png')
    plt.close()

    # Ratio of Runtimes between Algorithms per Number of Agents
    df_dmrsd = df[df['diagnosis algorithm'] == 'DMRSD_I1D1R1']
    df_parallel = df[df['diagnosis algorithm'] == 'parallel_DMRSD_I1D1R1']

    # Group by 'instance_number' and 'noa' and calculate the mean runtime for each algorithm
    df_dmrsd_mean = df_dmrsd.groupby(['instance_number', 'noa'])['Runtime'].mean().reset_index()
    df_parallel_mean = df_parallel.groupby(['instance_number', 'noa'])['Runtime'].mean().reset_index()

    # Merge the dataframes on 'instance_number' and 'noa'
    df_merged = pd.merge(df_dmrsd_mean, df_parallel_mean, on=['instance_number', 'noa'],
                         suffixes=('_dmrsd', '_parallel'))

    # Calculate the ratio of runtimes between algorithms per number of agents
    df_ratio = pd.DataFrame()
    df_ratio['instance_number'] = df_merged['instance_number']
    df_ratio['noa'] = df_merged['noa']
    df_ratio['Runtime Ratio'] = df_merged['Runtime_dmrsd'] / df_merged['Runtime_parallel']

    # Group by 'noa' and calculate the mean runtime ratio for each number of agents
    df_ratio_mean = df_ratio.groupby('noa')['Runtime Ratio'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(df_ratio_mean['noa'], df_ratio_mean['Runtime Ratio'])
    plt.title('Ratio of Runtimes between Algorithms per Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Average Runtime Ratio (DMRSD_I1D1R1 / parallel_DMRSD_I1D1R1)')
    plt.savefig('runtime_comparison_graphs/runtime_ratio_by_noa_grouped.png')
    plt.close()

    # Runtime Comparison by Instance Number (instance_number)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='instance_number', y='Runtime', hue='diagnosis algorithm')
    plt.title('Runtime Comparison by Instance Number')
    plt.xlabel('Instance Number')
    plt.ylabel('Runtime (seconds)')
    plt.legend(title='Diagnosis Algorithm')
    plt.savefig('runtime_comparison_graphs/runtime_by_instance.png')
    plt.close()

    # Apply log2 transformation to 'Runtime' for 'DMRSD_I1D1R1' algorithm
    df['Runtime_log2'] = np.where(df['diagnosis algorithm'] == 'DMRSD_I1D1R1', np.log2(df['Runtime']), df['Runtime'])

    # Separate DataFrames for each algorithm
    # df_dmrsd = df[df['diagnosis algorithm'] == 'DMRSD_I1D1R1']
    # df['Runtime_log2'] = np.where(df['diagnosis algorithm'] == 'parallel_DMRSD_I1D1R1', df['Runtime'], df['Runtime'])

    # Plot the combined runtime comparison using Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='instance_number', y='Runtime_log2', hue='diagnosis algorithm')
    plt.title('Runtime Comparison by Instance Number')
    plt.xlabel('Instance Number')
    plt.ylabel('Log2 of Runtime (seconds)')
    plt.legend(title='Diagnosis Algorithm')
    plt.savefig('runtime_comparison_graphs/runtime_log2_by_instance.png')
    plt.close()


    # Box Plots for Runtime Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='diagnosis algorithm', y='Runtime')
    plt.title('Box Plot of Runtime Distribution')
    plt.xlabel('Diagnosis Algorithm')
    plt.ylabel('Runtime (seconds)')
    plt.savefig('runtime_comparison_graphs/runtime_boxplot.png')
    plt.close()

    # Scatter Plot with Regression Line by Number of Agents (noa)
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='noa', y='Runtime', scatter=True, line_kws={"color": "red"})
    plt.title('Scatter Plot with Regression Line by Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Runtime (seconds)')
    plt.savefig('runtime_comparison_graphs/runtime_scatter_regression.png')
    plt.close()

    # Generate graphs
    # Runtime Comparison by Number of Agents (noa)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='noa', y='Runtime', hue='diagnosis algorithm')
    plt.title('Runtime Comparison by Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Runtime (seconds)')
    plt.legend(title='Diagnosis Algorithm')
    plt.savefig('runtime_comparison_graphs/runtime_by_noa.png')
    plt.close()

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='noa', y='Runtime_log2', hue='diagnosis algorithm')
    plt.title('Runtime Comparison by Number of Agents (log2 Transformation for DMRSD_I1D1R1)')
    plt.xlabel('Number of Agents')
    plt.ylabel('Runtime (log2 seconds)')
    plt.legend(title='Diagnosis Algorithm')
    plt.savefig('runtime_comparison_graphs/runtime_log2_by_noa.png')
    plt.close()


    # Calculate the log2 of Runtime for DMRSD_I1D1R1 algorithm
    # df_dmrsd['Runtime'] = np.log2(df_dmrsd['Runtime'])

    # Merge the two DataFrames based on 'noa'
    df_diff = pd.merge(df_dmrsd[['noa', 'Runtime']], df_parallel[['noa', 'Runtime']], on='noa',
                       suffixes=('_dmrsd', '_parallel'))

    # Calculate the difference between the log2 runtimes
    df_diff['Ratio'] = np.log2(df_diff['Runtime_dmrsd'] / df_diff['Runtime_parallel'])

    # Plot the difference using Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_diff, x='noa', y='Ratio')
    plt.axhline(y=0, color='red', linestyle='--')  # Add a horizontal line at y=0 for reference
    plt.title('Runtime ratio (log2) between DMRSD_I1D1R1 and parallel_DMRSD_I1D1R1')
    plt.xlabel('Number of Agents')
    plt.ylabel('Ratio in Runtime (log2 seconds)')
    plt.savefig('runtime_comparison_graphs/runtime_log2_ratio_by_noa.png')
    plt.close()

    # Perform T-Test
    algorithm1_data = df[df['diagnosis algorithm'] == 'DMRSD_I1D1R1']['Runtime']
    algorithm2_data = df[df['diagnosis algorithm'] == 'parallel_DMRSD_I1D1R1']['Runtime']
    t_stat, p_value = ttest_ind(algorithm1_data, algorithm2_data)

    df_avg_runtime = df.groupby(['noa', 'diagnosis algorithm'])['Runtime'].mean().reset_index()
    ratios = df_avg_runtime[['diagnosis algorithm'] == 'DMRSD_I1D1R1']['Runtime'] / df_avg_runtime[['diagnosis algorithm'] == 'parallel_DMRSD_I1D1R1']['Runtime']
    df_avg_runtime['Ratio'] = np.where(df_avg_runtime['diagnosis algorithm'] == 'DMRSD_I1D1R1', ratios, None)
    # Save the result to a CSV file
    df_avg_runtime.to_csv('average_runtime_by_noa_per_algorithm.csv', index=False)

    print("T-Test Results:")
    print(f"T-Statistic: {t_stat}")
    print(f"P-Value: {p_value}")

    print("All graphs saved in the 'runtime_comparison_graphs' folder.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    generate_graphs(csv_file)
