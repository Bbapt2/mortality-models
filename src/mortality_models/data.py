import pandas as pd
import numpy as np
from mortality_models.params import *

def prepare_data(D, E):
    """
    Function that prepares the data and puts it in the right format for mortality modeling.
    
    Parameters
    ----------
    D: pandas.DataFrame
        Dataframe with deaths. Columns must be years (e.g. [1950, 1951, ...]) and index must be ages (e.g. [50, 51, 52, ...]).
    E: pandas.DataFrame
        Dataframe with exposure. Columns must be years (e.g. [1950, 1951, ...]) and index must be ages (e.g. [50, 51, 52, ...]).
        
    Returns
    -------
    pandas.DataFrame
        Data prepared i.e. DataFrame with columns [YEAR, AGE, DX, EX, MX].
    """
    ages = np.intersect1d(E.index, D.index)
    years = np.intersect1d(E.columns, D.columns)
    data = pd.DataFrame()
    temp_D = D[D.index.isin(ages)][years].copy()
    temp_E = E[E.index.isin(ages)][years].copy()
    temp_mu = temp_D / temp_E
    temp_mu.fillna(0, inplace=True)
    for an in years:
        temp = pd.DataFrame(columns=[COLUMN_AGE, COLUMN_MX, COLUMN_YEAR])
        temp[COLUMN_MX] = temp_mu.loc[ages, an]
        temp[COLUMN_DX] = temp_D.loc[ages, an]
        temp[COLUMN_EX] = temp_E.loc[ages, an]
        temp[COLUMN_AGE] = ages
        temp[COLUMN_YEAR] = an
        if data.empty:
            data = temp
        else:
            data = pd.concat([data, temp])
    data[COLUMN_YEAR] = data[COLUMN_YEAR].astype(int)
    return data

class DataEdit: 
    """
    Class that edits the data (prepared beforehand) for the calibration.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Mortality data prepared: DataFrame with columns [YEAR, AGE, DX, EX, MX].
    min_year: int
        Minimum year considered for the calibration.
    max_year: int
        Maximum year considered for the calibration.
    min_age: int
        Minimum age considered for the calibration.
    max_age: int
        Maximum age considered for the calibration.
    
    Attributes
    ----------
    data_calibration: pandas.DataFrame
        Mortality data edited.
    """
    
    def __init__(self, data, min_year, max_year, min_age, max_age):
        self.data = data
        self.min_year = min_year
        self.max_year = max_year
        self.min_age = min_age
        self.max_age = max_age
        
    @property
    def data_calibration(self):
        """
        Function that prepares data for calibration by restrincting data to ages and years considered and adding some columns (GENERATION, LOG_MX, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES, ZEROS).
        
        Returns
        -------
        pandas.DataFrame
            Mortality data for calibration.
        """
        cond_age = self.data[COLUMN_AGE].isin(range(self.min_age, self.max_age+1))
        cond_year = self.data[COLUMN_YEAR].isin(range(self.min_year, self.max_year+1))
        data_calibration = self.data[cond_age & cond_year].copy()
        data_calibration[COLUMN_GENERATION] = data_calibration[COLUMN_YEAR] - data_calibration[COLUMN_AGE]
        data_calibration[COLUMN_LOG_MX] = data_calibration[COLUMN_MX].map(lambda x: np.log(max(x, EPS)))
        data_calibration[COLUMN_CENTERED_AGE] = data_calibration[COLUMN_AGE] - data_calibration[COLUMN_AGE].mean()
        temp = data_calibration[COLUMN_CENTERED_AGE] ** 2
        data_calibration[COLUMN_SIGMA_CENTERED_AGE] = temp - temp.mean()
        data_calibration[COLUMN_ONES] = 1
        data_calibration[COLUMN_ZEROS] = 0
        return data_calibration
    
    def edit_data(self, params):
        """
        Function that adds parameters to the calibration data.
        
        Parameters
        ----------
        params: dict
            Parameters to add to the calibration data. 
            These parameters are the alpha_x, beta_x, kappa_t and gamma_tx of a particular mortality model.
            They must be stored inside a dictionary of the form
            {PARAM1: {
                VALUE: float
                    Values of PARAM1.
                INDEX: float
                    Values of variable associated with PARAM1 (e.g. ages, years or generations).
                A1: list of str
                    List of columns names so that data[MX] = exp(data[A1[0]] * data[[A2[0]]] + data[A1[1]] * data[A2[1]] + ...).
                A2: list of str
                    List of columns names so that data[MX] = exp(data[A1[0]] * data[[A2[0]]] + data[A1[1]] * data[A2[1]] + ...).
                DERIVATIVE: str
                    Column name of derivative variable associated with PARAM1.
                VAR: str
                    Column name of variable associated with PARAM1 (e.g. AGE, YEAR or GENERATION).
                VALUE_NAME: str
                    Column name of parameter PARAM1.
                },
             ...
            }
            
        Returns
        -------
        pandas.DataFrame
            Mortality data with added columns.
        """
        data = self.data_calibration.copy()
        for p in params.keys():
            params_merge = pd.DataFrame(data=[params[p][PARAMS_INDEX], params[p][PARAMS_VALUE]], index=[params[p][PARAMS_VARIABLE], params[p][PARAMS_VALUE_NAME]]).T
            data = data.merge(params_merge)
        return data    