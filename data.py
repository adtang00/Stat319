from ucimlrepo import fetch_ucirepo 
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot
import statsmodels.api as sm
import statsmodels.formula.api as smf


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
    return runs

def bootstrap_qq(vals):
    stats.probplot(vals, dist="norm", plot=plt)
    plt.title('Q-Q Plot for Bootstrap Means of pH')
    plt.show()

def bootstrap_ci(boot_vals, ci=0.95):
    data = X['pH'].values
    n = len(data)
    boot_se = np.std(boot_vals, ddof=1)
    xbar= np.mean(data)
    t_crit = stats.t.ppf(1 - (1-ci) / 2, df=n - 1)
    lower = xbar - t_crit * boot_se
    upper = xbar + t_crit * boot_se

    print("Bootstrap t 95% CI:", lower, upper)


def mult_comparisons():
    from ucimlrepo import fetch_ucirepo 
    
    # fetch dataset 
    estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544) 
    
    # data (as pandas dataframes) 
    X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
    y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets 
    
    # metadata 
    print(estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.metadata) 
    
    # variable information 
    print(estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.variables) 

    df = pd.concat([X, y], axis=1)
    df.columns = list(X.columns) + list(y.columns)

    df['MTRANS_str'] = df['MTRANS'].astype(str)
    df['CAEC_str'] = df['CAEC'].astype(str)

    # 3️. Interaction plot
    plt.figure(figsize=(8,6))
    interaction_plot(df['MTRANS'], df['CAEC'], df['Weight'],
                     colors=['red','blue','green','orange'], markers=['o','s','^','D'], ms=8)
    plt.xlabel('MTRANS')
    plt.ylabel('Weight(kg)')
    plt.title('Interaction Plot: Weight by MTRANS and CAEC')
    plt.show()

    model = smf.ols("Weight ~ C(MTRANS) + C(CAEC)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    residuals = model.resid
    fitted = model.fittedvalues
    # Create plots
    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    # 1. Q-Q plot of residuals
    sm.qqplot(residuals, line='s', ax=ax[0])
    ax[0].set_title('Q-Q Plot of Residuals')

    # 2. Residuals vs Fitted plot
    ax[1].scatter(fitted, residuals)
    ax[1].axhline(0, color='red', linestyle='--')
    ax[1].set_xlabel('Fitted values')
    ax[1].set_ylabel('Residuals')
    ax[1].set_title('Residuals vs Fitted')

    plt.tight_layout()
    plt.show()

    # 4. Hypothesis test for equality of means
    # Calculate MSA and MSB
    msa = anova_table['sum_sq']['C(MTRANS)'] / anova_table['df']['C(MTRANS)']
    msb = anova_table['sum_sq']['C(CAEC)'] / anova_table['df']['C(CAEC)']
    mse = anova_table['sum_sq']['Residual'] / anova_table['df']['Residual']

    # Calculate F-statistics
    f_stat_a = msa / mse
    f_stat_b = msb / mse
    print(f"F-statistic for MTRANS: {f_stat_a}")
    print(f"F-statistic for CAEC: {f_stat_b}")

    # Critical F-value
    alpha = 0.05
    f_crit_a = stats.f.ppf(1 - alpha, anova_table['df']['C(MTRANS)'], anova_table['df']['Residual'])
    f_crit_b = stats.f.ppf(1 - alpha, anova_table['df']['C(CAEC)'], anova_table['df']['Residual'])
    print(f"Critical F-value for MTRANS: {f_crit_a}")
    print(f"Critical F-value for CAEC: {f_crit_b}")

    # p-values
    p_value_a = 1 - stats.f.cdf(f_stat_a, anova_table['df']['C(MTRANS)'], anova_table['df']['Residual'])
    p_value_b = 1 - stats.f.cdf(f_stat_b, anova_table['df']['C(CAEC)'], anova_table['df']['Residual'])
    print(f"P-value for MTRANS: {p_value_a}")   
    print(f"P-value for CAEC: {p_value_b}")

    # Conclusion
    if f_stat_a > f_crit_a:
        print("Reject null hypothesis for MTRANS: At least one group mean is different.")
    else:
        print("Fail to reject null hypothesis for MTRANS: No significant difference between group means.") 
    if f_stat_b > f_crit_b:
        print("Reject null hypothesis for CAEC: At least one group mean is different.")
    else:
        print("Fail to reject null hypothesis for CAEC: No significant difference between group means.")

if __name__ == "__main__":
    #bootstrap_vals = bootstrap()
    #bootstrap_qq(bootstrap_vals)
    #bootstrap_ci(bootstrap_vals)
    mult_comparisons()
    pass