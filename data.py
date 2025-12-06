from ucimlrepo import fetch_ucirepo 
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
#print(wine_quality.metadata) 
  
# variable information 
#print(wine_quality.variables) 

# Number of rows and columns in X
#print("Number of rows:", X.shape[0])
#print("Number of columns:", X.shape[1])

# Number of rows in y (targets)
#print("Number of target rows:", y.shape[0])

#print("mean alcohol", X.iloc[:1000]['alcohol'].mean())

def summary_statistics():
    # Show all columns and widen display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(X.describe())
    

def plot_histogram_q1():
    data = X["density"].values

    sample_size = 50
    num_samples = 5000

    sample_means = []

    for i in range(num_samples):
        sample = np.random.choice(data, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))

    plt.hist(sample_means, bins=30, edgecolor='black')
    plt.xlabel("Sample Mean of Density")
    plt.title("Sampling Distribution of Sample Mean (Density)")
    plt.show()


def ttest():
    # Sample data
    sample_data = X['alcohol'].values

    # Hypothesized population mean
    pop_mean = 10.50

    # Sample statistics
    n = len(sample_data)
    df = n - 1
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)

    # Compute t-statistic manually
    t_statistic = (sample_mean - pop_mean) / (sample_std / np.sqrt(n))

    # Find t-critical value for two-tailed test
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha/2, df)

    # Print results
    print(f"Sample mean: {sample_mean:.4f}")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"T-critical (two-tailed, alpha={alpha}): ±{t_crit:.4f}")

    # Decision based on t-critical
    if abs(t_statistic) > t_crit:
        print("Reject the null hypothesis based on t-critical value.")
    else:
        print("Fail to reject the null hypothesis based on t-critical value.")

    # Also show p-value for reference
    t_stat, p_value = stats.ttest_1samp(a=sample_data, popmean=pop_mean)
    print(f"P-value: {p_value:.4f}")
    if p_value < alpha:
        print("Reject the null hypothesis based on p-value.")
    else:
        print("Fail to reject the null hypothesis based on p-value.")

def bootstrap():
    # Run bootstrap on ph values
    df = X['pH'].values

    iter = 10000
    runs = np.zeros(shape = (iter))

    for i in range(iter):
        sample = np.random.choice(df, size = len(df), replace=True)
        runs[i,] = np.mean(sample)
    plt.hist(x = runs, bins = 'auto', alpha = 0.7)
    plt.xlabel('Mean of ph')
    plt.ylabel('Frequency')
    plt.title('Bootstrap Distribution of Mean ph')
    plt.show()

if __name__ == "__main__":
    bootstrap()
    pass