import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('clean_data/data_clean_weighted.csv')
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Sex distribution:\n{df['sex'].value_counts()}")
print(f"ever_married distribution:\n{df['ever_married'].value_counts()}")
print(f"is_divorced distribution:\n{df['is_divorced'].value_counts()}")

def run_weighted_model(df_sub, outcome, formula_rhs, label):
    pred_cols = [c.strip() for c in formula_rhs.replace('~','').split('+')]
    df_m = df_sub.dropna(subset=[outcome] + pred_cols + ['pwt14xa']).copy()
    n = len(df_m)
    w = df_m['pwt14xa'].values
    w_norm = w / w.sum() * n
    y, X = patsy.dmatrices(f'{outcome} {formula_rhs}', data=df_m, return_type='dataframe')
    model = sm.Logit(y.values.ravel(), X, freq_weights=w_norm).fit(disp=False)
    pr2 = 1 - model.llf / model.llnull
    print(f"\n{'='*60}")
    print(f"MODEL: {label}")
    print(f"N = {n}, Pseudo R2 = {pr2:.4f}")
    print(f"{'='*60}")
    results = pd.DataFrame({
        'coef': model.params,
        'se': model.bse,
        'z': model.tvalues,
        'p': model.pvalues,
        'OR': np.exp(model.params),
        'CI_lo': np.exp(model.conf_int()[0]),
        'CI_hi': np.exp(model.conf_int()[1]),
    })
    print(results.round(4).to_string())
    print(f"\nLog-likelihood: {model.llf:.2f}, Null LL: {model.llnull:.2f}, Pseudo R2: {pr2:.4f}")
    return model

formula_base  = '~ log_income + education + age + age_squared + is_jawa + is_sunda'
formula_urban = '~ log_income + education + age + age_squared + is_jawa + is_sunda + urban'

df_women = df[df['sex'] == 'Female'].copy()
df_men   = df[df['sex'] == 'Male'].copy()

print(f"\nWomen N={len(df_women)}, Men N={len(df_men)}")

# Ever Married
m1 = run_weighted_model(df_women, 'ever_married', formula_base,  'Women: Ever Married (Base)')
m2 = run_weighted_model(df_men,   'ever_married', formula_base,  'Men: Ever Married (Base)')
m3 = run_weighted_model(df_women, 'ever_married', formula_urban, 'Women: Ever Married (Urban)')
m4 = run_weighted_model(df_men,   'ever_married', formula_urban, 'Men: Ever Married (Urban)')

# Ever Divorced
m5 = run_weighted_model(df_women, 'is_divorced', formula_base,  'Women: Ever Divorced (Base)')
m6 = run_weighted_model(df_men,   'is_divorced', formula_base,  'Men: Ever Divorced (Base)')
m7 = run_weighted_model(df_women, 'is_divorced', formula_urban, 'Women: Ever Divorced (Urban)')
m8 = run_weighted_model(df_men,   'is_divorced', formula_urban, 'Men: Ever Divorced (Urban)')

print("\n\nDONE")
