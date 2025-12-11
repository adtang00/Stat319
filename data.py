from ucimlrepo import fetch_ucirepo 
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fetch the Wine Quality dataset
wine_quality = fetch_ucirepo(id=186)

# Features and targets as pandas DataFrames
X = wine_quality.data.features
y = wine_quality.data.targets

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

def bootstrap():
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features
    df = X['pH'].values

    iter = 10000
    runs = np.zeros(iter)

    for i in range(iter):
        sample = np.random.choice(df, size=len(df), replace=True)
        runs[i] = np.mean(sample)
    return runs

def plot_bootstrap(runs):
    plt.hist(x = runs, bins = 'auto', alpha = 0.7)
    plt.xlabel('Mean of ph')
    plt.ylabel('Frequency')
    plt.title('Bootstrap Distribution of Mean ph')
    plt.show()

def bootstrap_qq(vals):
    stats.probplot(vals, dist="norm", plot=plt)
    plt.title("QQ Plot for Bootstrap Means of pH")
    plt.show()

def interaction_plot_model():
    # fetch dataset 
    estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544) 
    
    # data (as pandas dataframes) 
    X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
    y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets 
    
    # metadata 
    #print(estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.metadata) 
    
    # variable information 
    #print(estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.variables) 

    df = pd.concat([X, y], axis=1)
    df.columns = list(X.columns) + list(y.columns)

    df['MTRANS_str'] = df['MTRANS'].astype(str)
    df['CAEC_str'] = df['CAEC'].astype(str)


    # 3️. Interaction plot
    plt.figure(figsize=(12,8))
    interaction_plot(df['MTRANS'], df['CAEC'], df['Weight'],
                     colors=['red','blue','green','orange'], markers=['o','s','^','D'], ms=8)
    plt.xlabel('MTRANS')
    plt.ylabel('Weight(kg)')
    plt.title('Interaction Plot: Weight by MTRANS and CAEC')
    plt.show()

    model = smf.ols("Weight ~ C(MTRANS) + C(CAEC)", data=df).fit()
    #anova_table = sm.stats.anova_lm(model, typ=2)
    #print(anova_table)
    return model

def plot_resid_fitted(model):
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

def hypothesis_test(model):
    anova_table = sm.stats.anova_lm(model, typ=2)

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

    return anova_table

def multiple_comparisons(model):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import pandas as pd

    # Extract dataframe used in the model
    df = model.model.data.frame.copy()

    # Convert factors to string for Tukey labels
    df['MTRANS_str'] = df['MTRANS'].astype(str)
    df['CAEC_str'] = df['CAEC'].astype(str)

    # --- Tukey for MTRANS ---
    tukey_mtrans = pairwise_tukeyhsd(
        endog=df['Weight'],
        groups=df['MTRANS_str'],
        alpha=0.05
    )

    # Convert Tukey result to DataFrame
    mtrans_df = pd.DataFrame(
        data=tukey_mtrans._results_table.data[1:],   # skip header row
        columns=tukey_mtrans._results_table.data[0]  # use header row
    )

    print("\n=== Tukey HSD for MTRANS ===")
    print(mtrans_df.to_markdown(index=False))


    # --- Tukey for CAEC ---
    tukey_caec = pairwise_tukeyhsd(
        endog=df['Weight'],
        groups=df['CAEC_str'],
        alpha=0.05
    )

    caec_df = pd.DataFrame(
        data=tukey_caec._results_table.data[1:],
        columns=tukey_caec._results_table.data[0]
    )

    print("\n=== Tukey HSD for CAEC ===")
    print(caec_df.to_markdown(index=False))

    return mtrans_df, caec_df


if __name__ == "__main__":
    model = interaction_plot_model()
    multiple_comparisons(model)
    pass