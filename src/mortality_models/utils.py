import pandas as pd
import numpy as np
from mortality_models.params import *
from mortality_models.data import DataEdit

class MortalityModel(DataEdit):
    """
    Class for mortality models.
    Model has the form:
        eta(x, t) = a(x) + sum_{i}(b_i(x) * k_i(t)) + b(0) * g(t, x)
    Newton-Raphson algorithm is applied to calculate the parameters.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Mortality data considered. 
        DataFrame with columns [YEAR, AGE, DX, EX, MX, GENERATION, LOG_MX, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES, ZEROS].
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
    x: range
        Ages considered for the modeling.
    t: range
        Years considered for the modeling.
    g: range
        Generations considered for the modeling.
    data_calibration: pandas.DataFrame
        Data used for the calibration of the model, transformed according to needs.
    n_iter: int
        Number of iterations necessary for the Newton-Raphson algorithm to converge.
    gap: float
        Quadratic difference of parameters between the two last iterations of the algorithm. 
    convergence: bool
        Whether the Newwton-Raphson algorithm converged or not.
    """
    
    def __init__(self, data, min_year, max_year, min_age, max_age):
        super().__init__(data, min_year, max_year, min_age, max_age)
        self.x = range(min_age, max_age+1)
        self.t = range(min_year, max_year+1)
        self.g = range((min_year - max_age), (max_year - min_age)+1)
    
    def l1(self, data, a1, a2, derivative, col):
        """
        Calculate the derivative of the log-likelihook for the Poisson model with respect to a given parameter.
        
        Parameters
        ----------
        data : pd.DataFrame
            Mortality data considered. 
            DataFrame with columns [YEAR, AGE, DX, EX, MX, GENERATION, LOG_MX, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES, ZEROS].
        a1 : list of str
            List of columns names so that data[MX] = exp(data[a1[0]] * data[[a2[0]]] + data[a1[1]] * data[a2[1]] + ...).
        a2 : list of str
            List of columns names so that data[MX] = exp(data[a1[0]] * data[[a2[0]]] + data[a1[1]] * data[a2[1]] + ...).
        derivative : str
            Column name of derivative variable.
        col : str
            Column name of variable associated (e.g. AGE, YEAR or GENERATION).
        
        Returns
        -------
        np.array
            Value of the derivative of the log-likelihood for each variable (age, year or generation).
        """
        temp = data.copy()
        eta = pd.Series(np.zeros(len(temp)))
        for i, j in enumerate(a1):
            eta += temp[a1[i]] * temp[a2[i]]
        d_eta = temp[derivative]
        temp[COLUMN_RESULT] = (temp[COLUMN_DX] - temp[COLUMN_EX] * np.exp(eta)) * d_eta
        return np.array(temp.groupby([col]).aggregate({COLUMN_RESULT: "sum"})).flatten()
    
    def dl1(self, data, a1, a2, derivative, col):
        """
        Calculate the second derivative of the log-likelihook for the Poisson model with respect to a given parameter.
        
        Parameters
        ----------
        data : pd.DataFrame
            Mortality data considered. 
            DataFrame with columns [YEAR, AGE, DX, EX, MX, GENERATION, LOG_MX, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES, ZEROS].
        a1 : list of str
            List of columns names so that data[MX] = exp(data[a1[0]] * data[[a2[0]]] + data[a1[1]] * data[a2[1]] + ...).
        a2 : list of str
            List of columns names so that data[MX] = exp(data[a1[0]] * data[[a2[0]]] + data[a1[1]] * data[a2[1]] + ...).
        derivative : str
            Column name of derivative variable.
        col : str
            Column name of variable associated (e.g. AGE, YEAR or GENERATION).
        
        Returns
        -------
        np.array
            Value of the second derivative of the log-likelihood for each variable (age, year or generation).
        """
        temp = data.copy()
        eta = pd.Series(np.zeros(len(temp)))
        for i, j in enumerate(a1):
            eta += temp[a1[i]] * temp[a2[i]]
        d_eta = temp[derivative]
        temp[COLUMN_RESULT] = - (temp[COLUMN_EX] * np.exp(eta)) * d_eta**2
        return np.array(temp.groupby([col]).aggregate({COLUMN_RESULT: "sum"})).flatten()
    
    def constraints(self, params):
        """
        Define the constraints applied for identifiability of the model.
        
        Parameters
        ----------
        params : dict
            Parameters of the model. 
            Must be of the form
            {PARAM1: {
                VALUE: float
                    Values of PARAM1.
                INDEX: float
                    Values of variable associated with PARAM1 (e.g. ages, years or generations).
                A1: list 
                    List of columns names so that data[MX] = exp(data[A1[0]] * data[[A2[0]]] + data[A1[1]] * data[A2[1]] + ...).
                A2: list
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
        dict
            Parameters transformed to respect the identifiability constraints.
            Same format as params.
        """
        return params
    
    def fit_predict(self, data, a1, a2):
        temp = data.copy()
        eta = pd.Series(np.zeros(len(temp)))
        for i, j in enumerate(a1):
            eta += temp[a1[i]] * temp[a2[i]]
        temp[COLUMN_MX_PRED] = np.exp(eta)
        temp[COLUMN_DX_PRED] = temp[COLUMN_EX] * temp[COLUMN_MX_PRED]
        return temp
    
    def fit_model(self, params_init, tol, max_iter):
        """
        Calculate the best parameters for the model.
        
        Parameters
        ----------
        params_init : dict
            Parameters of the model. 
            Must be of the form 
            {PARAM1: {
                VALUE: float
                    Values of PARAM1.
                INDEX: float
                    Values of variable associated with PARAM1 (e.g. ages, years or generations).
                A1: list 
                    List of columns names so that data[MX] = exp(data[A1[0]] * data[[A2[0]]] + data[A1[1]] * data[A2[1]] + ...).
                A2: list
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
        tol : float
            Error tolerance for the Newton-Raphson algorithm to stop.
        max_iter : int
            Maximum number of iteration of the Newton-Raphson algorithm to perform.
        
        Returns
        -------
        dict
            Best parameters for the model. Same format as params_init.
        """
        params_f = params_init.copy()
        data = self.edit_data(params_init)
        loop = True
        i = 0
        while loop:
            if i > max_iter:
                break
            else:
                loop_temp = 0
                for p in params_init.keys():
                    p_cible = params_f[p]
                    p_cible_init = p_cible[PARAMS_VALUE]
                    a1 = p_cible[PARAMS_A1]
                    a2 = p_cible[PARAMS_A2]
                    derivative = p_cible[PARAMS_DERIVATIVE]
                    col = p_cible[PARAMS_VARIABLE]
                    index = p_cible[PARAMS_INDEX]
                    p_cible[PARAMS_VALUE] = p_cible[PARAMS_VALUE] - (self.l1(data, a1, a2, derivative, col) / self.dl1(data, a1, a2, derivative, col))
                    params_f[p] = p_cible
                    loop_temp += ((p_cible_init - p_cible[PARAMS_VALUE])**2).sum().sum()
                    data = self.edit_data(params_f)
                i += 1
                loop = loop_temp >= tol
        params_f = self.constraints(params_f)
        nu = sum([len(params_f[p][PARAMS_VALUE]) for p in params_f])
        data = self.edit_data(params_f)
        self.data_predict = self.fit_predict(data, a1, a2)
        self.response_resid = response_residuals(data=self.data_predict, min_year=self.min_year, max_year=self.max_year) 
        self.deviance_resid = deviance_residuals(data=self.data_predict, nu=nu, min_year=self.min_year, max_year=self.max_year)
        self.n_iter = i
        self.gap = loop_temp
        self.convergence = ~loop
        return params_f
    
def deviance(D, D_pred):
    """
    Function that calculates the deviance of a Poisson model.
    
    Parameters
    ----------
    D: pandas.Series
        Real deaths.
    D_pred: pandas.Series
        Predicted deaths.
        
    Returns
    -------
    pandas.Series
        Deviance for each (D, D_pred) observation.
    """
    return np.maximum(0, np.where(D==0, 2 * D_pred,  2 * (D * np.log(D / D_pred) - (D - D_pred))))

def response_residuals(data, min_year, max_year):
    """
    Function that calculates the response residuals.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Mortality data.
        DataFrame with columns [YEAR, AGE, DX, EX, MX, GENERATION, LOG_MX, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES, ZEROS, MX_PRED].
    min_year: int
        Minimum year considered for the calibration.
    max_year: int
        Maximum year considered for the calibration.
    
    Returns
    -------
    pandas.DataFrame
        Response residuals.
        Index is ages, columns are years.
    """
    data_temp = data.copy()
    data_temp[COLUMN_RESIDUALS] = data_temp[COLUMN_MX] - data_temp[COLUMN_MX_PRED]
    residuals = pd.DataFrame(columns=[COLUMN_AGE])
    for year in range(min_year, max_year + 1):
        temp = data_temp[data_temp[COLUMN_YEAR]==year][[COLUMN_AGE, COLUMN_RESIDUALS]].copy()
        temp.rename(columns={COLUMN_RESIDUALS: year}, inplace=True)
        residuals = residuals.merge(temp, on=[COLUMN_AGE], how='outer')
    residuals.set_index(COLUMN_AGE, inplace=True)
    return residuals

def deviance_residuals(data, nu, min_year, max_year):
    """
    Function that calculates the response residuals.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Mortality data.
        DataFrame with columns [YEAR, AGE, DX, EX, MX, GENERATION, LOG_MX, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES, ZEROS, MX_PRED].
    nu: int
        Effective number of parameters in the model.
    min_year: int
        Minimum year considered for the calibration.
    max_year: int
        Maximum year considered for the calibration.
    
    Returns
    -------
    pandas.DataFrame
        Deviance residuals.
        Index is ages, columns are years.
    """
    data_temp = data.copy()
    D = data_temp[COLUMN_DX]
    D_pred = data_temp[COLUMN_DX_PRED]
    dev = deviance(D, D_pred)
    K = data_temp[COLUMN_ONES].sum()
    phi = sum(dev) / (K - nu)
    sg = np.where(D-D_pred==0, 0, (D-D_pred)/(abs(D-D_pred)))
    data_temp[COLUMN_RESIDUALS] = sg * np.sqrt(dev / phi)
    residuals = pd.DataFrame(columns=[COLUMN_AGE])
    for year in range(min_year, max_year + 1):
        temp = data_temp[data_temp[COLUMN_YEAR]==year][[COLUMN_AGE, COLUMN_RESIDUALS]].copy()
        temp.rename(columns={COLUMN_RESIDUALS: year}, inplace=True)
        residuals = residuals.merge(temp, on=[COLUMN_AGE], how='outer')
    residuals.set_index(COLUMN_AGE, inplace=True)
    return residuals              