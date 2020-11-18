import numpy as np
import pandas as pd
import sys
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels
import statsmodels.api as sm
import bokeh
import matplotlib
import seaborn as sns

from typing import List

from constants import *


def print_versions() -> None:
    "This function prints versions of the main packages."

    print("python version " + sys.version)
    print('numpy version ' + np.__version__)
    print('pandas version ' + pd.__version__)
    print('sklearn version ' + sklearn.__version__)
    print('bokeh version ' + bokeh.__version__)
    print('statsmodels version ' + statsmodels.__version__)
    print("matplotlib version " + matplotlib.__version__)
    print("seaborn version " + sns.__version__)


def investigate_object(df: pd.DataFrame) -> None:
    """
    This function prints the unique categories of all the object dtype columns.
    It prints '...' if there are more than 13 unique categories.
    """
    col_obj = df.columns[df.dtypes == 'object']

    for col_obj_i in col_obj:
        if len(df[col_obj_i].unique()) > 13:
            print(f"{col_obj_i}: Unique Values: ",
                  np.append(df[col_obj_i].unique()[:13], "..."))
        else:
            print(f"{col_obj_i}: Unique Values: {df[col_obj_i].unique()}")

    del col_obj


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans data by removing unnecessary characters
    and converting columns x12 and x64 to float.
    """
    replacements_dicts = {'x12': {'$': '', ',': '', ')': '', '(': '-', },
                          'x63': {'%': ''}}

    # iterate through cols to clean in replacements_dict
    for col, value in replacements_dicts.items():
        # iterate through strings to replace
        for old_str, new_str in value.items():
            # replace strings
            df[col] = df[col].str.replace(old_str, new_str)
        # convert to float
        df[col] = df[col].astype(float)

    return df


def prep_dataset(df: pd.DataFrame, imputer, std_scaler,
                 imputer_method: str = 'fit_transform',
                 drop_cols: List[str] = ['x5', 'x31', 'x81', 'x82']
                 ) -> pd.DataFrame:
    """
    This function prepares the dataset by imputing, standardizing,
    and creating dummie variables.
    """
    cols = ['y'] + drop_cols if 'y' in df.columns else drop_cols
    # get mean imputation
    df_imputed = pd.DataFrame(
        getattr(imputer, imputer_method)(df.drop(columns=cols)),
        columns=df.drop(columns=cols).columns)

    # standardize
    df_imputed_std = pd.DataFrame(
        getattr(std_scaler, imputer_method)(df_imputed),
        columns=df_imputed.columns)

    # create dummies / one hot encoding
    for col in drop_cols:
        dummie = pd.get_dummies(df[col],
                                drop_first=True, prefix=col,
                                prefix_sep='_', dummy_na=True)
        df_imputed_std = pd.concat([df_imputed_std, dummie],
                                   axis=1, sort=False)
        del dummie

    if 'y' in df.columns:
        df_imputed_std = pd.concat([df_imputed_std, df['y']],
                                   axis=1, sort=False)

    return df_imputed_std


def fit_model(df: pd.DataFrame, variables: List[str]):
    "This function fits a logit model and returns the result."

    # create model
    logit = sm.Logit(df['y'], df[variables])
    # fit model
    result = logit.fit()

    return result


def select_features(df: pd.DataFrame, logistic_regression, n=25) -> List[str]:
    """
    This function selects features from a logistic regression model
    by choosing the n features with the largest squared coefficients.
    """
    logistic_regression.fit(df.drop(columns=['y']), df['y'])

    # get model results
    exploratory_results = pd.DataFrame(
        df.drop(columns=['y']).columns).rename(columns={0: 'name'})
    exploratory_results['coefs'] = logistic_regression.coef_[0]
    exploratory_results['coefs_squared'] = exploratory_results['coefs']**2

    # select n variables with largest squared coefficients
    var_reduced = exploratory_results.nlargest(n, 'coefs_squared')
    variables = var_reduced['name'].to_list()

    return variables


def get_outcomes(df: pd.DataFrame, result, variables: List[str]
                 ) -> pd.DataFrame:
    "This function gets model predictions on dataframe."

    # when running a single record at a time,
    # some of the dummies may not have been creaetd
    try:
        outcomes_df = pd.DataFrame(result.predict(df[variables]))
    except KeyError as e:
        # get list of columns that weren't created
        cols_str = str(e)[2:-16]
        cols = [col.replace("\'", "") for col in cols_str.split("\', \'")]
        # create the columns, setting them to 0.0
        for col in cols:
            df[col] = 0.0
        outcomes_df = pd.DataFrame(result.predict(df[variables]))

    outcomes_df = outcomes_df.rename(columns={0: 'probs'})

    if 'y' in df.columns:
        outcomes_df['y'] = df['y']

    return outcomes_df


def calculate_c_statistic(df: pd.DataFrame) -> np.float:
    "This function calculates the C-Statistic on a dataframe."

    return roc_auc_score(df['y'], df['probs'])


def create_probability_bins(df: pd.DataFrame,
                            group: bool = False) -> pd.DataFrame:
    "This function creates 20 probability bins for a dataframe."

    df['prob_bin'] = pd.qcut(df['probs'], q=20)

    if group:
        df = df.groupby(['prob_bin'])['y'].sum()

    return df


def get_cutoff(df: pd.DataFrame) -> np.float:
    """
    This function calculates the 75th percentile of probabilities,
    which is used as the cutoff value for an event.
    """
    return df['probs'].quantile(0.75)


def make_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function uses the cutoff value to determine
    the occurrence of an event.
    """
    df['business_outcome'] = np.where(df['probs'] > cutoff, 1, 0)
    df = df.rename({'probs': 'phat'}, axis=1)

    return df
