import numpy as np
from mortality_models.utils import MortalityModel
from mortality_models.params import *

class LeeCarter(MortalityModel):
    """
    Class for Lee-Carter model.
    Model has the form:
        eta(x, t) = a(x) + b_1(x) * k_1(t),
    with sum(beta(x)) = 1 and sum(kappa(t)) = 0.
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
        
    def edit_params(self, alpha_x, beta_x, kappa_t):
        """
        Create a dictionary from the input to fit a Lee-Carter model:
            eta(x, t) = alpha(x) + beta(x) * kappa(t),
        with sum(beta(x)) = 1 and sum(kappa(t)) = 0.
        
        Parameters
        ----------
        alpha_x: numpy.array
            alpha parameters of the Lee-Carter model.
        beta_x: numpy.array
            beta parameters of the Lee-Carter model.
        kappa_t: numpy.array
            kappa parameters of the Lee-Carter model.
            
        Returns
        -------
        dict
            Dictionary of the form 
            {ALPHA_X: {
                VALUE: alpha_x
                INDEX: x
                A1: [ALPHA_X, KAPPA1_T]
                A2: [ONES, BETA1_X]
                DERIVATIVE: ONES
                VAR: AGE
                VALUE_NAME: ALPHA_X
                },
             BETA1_X: {
                VALUE: beta_x
                INDEX: x
                A1: [ALPHA_X, KAPPA1_T]
                A2: [ONES, BETA1_X]
                DERIVATIVE: KAPPA1_T
                VAR: AGE
                VALUE_NAME: BETA1_X
                },
             KAPPA1_T: {
                VALUE: kappa_t
                INDEX: t
                A1: [ALPHA_X, KAPPA1_T]
                A2: [ONES, BETA1_X]
                DERIVATIVE: BETA1_X
                VAR: YEAR
                VALUE_NAME: ALKAPPA1_TPHA_X
                }
            }
        """
        A1 = [LC_AX, LC_K1T]
        A2 = [COLUMN_ONES, LC_B1X]
        params = {
            LC_AX: {
                PARAMS_VALUE: alpha_x, 
                PARAMS_INDEX: self.x, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_ONES, 
                PARAMS_VARIABLE: COLUMN_AGE,
                PARAMS_VALUE_NAME: LC_AX
            },
            LC_B1X: {
                PARAMS_VALUE: beta_x, 
                PARAMS_INDEX: self.x, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: LC_K1T,
                PARAMS_VARIABLE: COLUMN_AGE, 
                PARAMS_VALUE_NAME: LC_B1X
            }, 
            LC_K1T: {
                PARAMS_VALUE: kappa_t, 
                PARAMS_INDEX: self.t, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: LC_B1X, 
                PARAMS_VARIABLE: COLUMN_YEAR, 
                PARAMS_VALUE_NAME: LC_K1T
            }
        }
        return params
        
    def constraints(self, params):
        """
        Overides MortalityModel.constraints to adjust for Lee-Carter constraints i.e. sum(beta(x)) = 1 and sum(kappa(t)) = 0.
        
        Parameters
        ----------
        params: dict
            Parameters of the model with the following form 
            {ALPHA_X: {
                VALUE: alpha_x
                INDEX: x
                A1: [ALPHA_X, KAPPA1_T]
                A2: [ONES, BETA1_X]
                DERIVATIVE: ONES
                VAR: AGE
                VALUE_NAME: ALPHA_X
                },
             BETA1_X: {
                VALUE: beta_x
                INDEX: x
                A1: [ALPHA_X, KAPPA1_T]
                A2: [ONES, BETA1_X]
                DERIVATIVE: KAPPA1_T
                VAR: AGE
                VALUE_NAME: BETA1_X
                },
             KAPPA1_T: {
                VALUE: kappa_t
                INDEX: t
                A1: [ALPHA_X, KAPPA1_T]
                A2: [ONES, BETA1_X]
                DERIVATIVE: BETA1_X
                VAR: YEAR
                VALUE_NAME: KAPPA1_T
                }
            }
            
        Returns 
        -------
        dict
            Parameters adjusted to fit the constraints. Same form as the input.
        """
        params_f = params.copy()
        c1 = np.mean(params[LC_K1T][PARAMS_VALUE])
        c2 = np.sum(params[LC_B1X][PARAMS_VALUE])
        params_f[LC_AX][PARAMS_VALUE] = params_f[LC_AX][PARAMS_VALUE] + c1 * params_f[LC_B1X][PARAMS_VALUE]
        params_f[LC_B1X][PARAMS_VALUE] = params_f[LC_B1X][PARAMS_VALUE] / c2
        params_f[LC_K1T][PARAMS_VALUE] = c2 * (params_f[LC_K1T][PARAMS_VALUE] - c1)
        return params_f
    
    def fit(self, alpha_x, beta_x, kappa_t, tol=EPS, max_iter=MAX_ITER):
        """
        Fits Lee-Carter model.
        
        Parameters
        ----------
        alpha_x: numpy.array
            Initial values for alpha_x.
        beta_x: numpy.array
            Initial values for beta_x.
        kappa_t: numpy.array
            Initial values for kappa_t.
        tol: float
            Error tolerance for the Newton-Raphson algorithm to stop.
        max_iter: int
            Maximum number of iteration of the Newton-Raphson algorithm performed.
        
        Returns
        -------
        numpy.array
            alpha_x fitted.
        numpy.array
            beta_x fitted.
        numpy.array
            kappa_t fitted.
        """
        params_init = self.edit_params(alpha_x, beta_x, kappa_t)
        params_f = self.fit_model(params_init, tol, max_iter)
        alpha_x_f = params_f[LC_AX][PARAMS_VALUE]
        beta_x_f = params_f[LC_B1X][PARAMS_VALUE]
        kappa_t_f = params_f[LC_K1T][PARAMS_VALUE]
        return alpha_x_f, beta_x_f, kappa_t_f

class CBD(MortalityModel):
    """
    Class for Cairns-Blake-Dowd model.
    Model has the form:
        eta(x, t) = kappa_1(t) + (x - x_bar) * kappa_2(t),
    where x_bar is the average age in the data.
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
        
    def edit_params(self, kappa1_t, kappa2_t):
        """
        Create a dictionary from the input to fit a Cairns-Blake-Dowd model:
            eta(x, t) = kappa_1(t) + (x - x_bar) * kappa_2(t),
        where x_bar is the average age in the data.
        
        Parameters
        ----------
        kappa1_t: numpy.array
            kappa_1 parameters of the Cairns-Blake-Dowd model.
        kappa1_t: numpy.array
            kappa_2 parameters of the Cairns-Blake-Dowd model.
            
        Returns
        -------
        dict
            Dictionary of the form 
            {KAPPA1_T: {
                VALUE: kappa1_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T]
                A2: [ONES, CENTERED_AGE]
                DERIVATIVE: ONES
                VAR: YEAR
                VALUE_NAME: KAPPA1_T
                },
             KAPPA2_T: {
                VALUE: kappa2_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T]
                A2: [ONES, CENTERED_AGE]
                DERIVATIVE: CENTERED_AGE
                VAR: YEAR
                VALUE_NAME: KAPPA2_T
                }
            }
        """
        A1 = [CBD_K1T, CBD_K2T]
        A2 = [COLUMN_ONES, COLUMN_CENTERED_AGE]
        params = {
            CBD_K1T: {
                PARAMS_VALUE: kappa1_t, 
                PARAMS_INDEX: self.t, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_ONES, 
                PARAMS_VARIABLE: COLUMN_YEAR, 
                PARAMS_VALUE_NAME: CBD_K1T
            },
            CBD_K2T: {
                PARAMS_VALUE: kappa2_t, 
                PARAMS_INDEX: self.t, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_CENTERED_AGE, 
                PARAMS_VARIABLE: COLUMN_YEAR, 
                PARAMS_VALUE_NAME: CBD_K2T
            }
        }
        return params
    
    def fit(self, kappa1_t, kappa2_t, tol=EPS, max_iter=MAX_ITER):
        """
        Fits Cairns-Blake-Dowd model.
        
        Parameters
        ----------
        kappa1_t: numpy.array
            Initial values for kappa1_t.
        kappa2_t: numpy.array
            Initial values for kappa2_t.
        tol: float
            Error tolerance for the Newton-Raphson algorithm to stop.
        max_iter: int
            Maximum number of iteration of the Newton-Raphson algorithm performed.
        
        Returns
        -------
        numpy.array
            kappa1_t fitted.
        numpy.array
            kappa2_t fitted.
        """
        params_init = self.edit_params(kappa1_t, kappa2_t)
        params_f = self.fit_model(params_init, tol, max_iter)
        kappa1_t_f = params_f[CBD_K1T][PARAMS_VALUE]
        kappa2_t_f = params_f[CBD_K2T][PARAMS_VALUE]
        return kappa1_t_f, kappa2_t_f

class M6(MortalityModel):
    """
    Class for M6 model.
    Model has the form:
        eta(x, t) = kappa_1(t) + (x - x_bar) * kappa_2(t) + g(t, x),
    where x_bar is the average age in the data, and sum(g(c))=0 and sum(cg(c)) = 0.
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
        
    def edit_params(self, kappa1_t, kappa2_t, gamma_tx):
        """
        Create a dictionary from the input to fit a M6 model:
            eta(x, t) = kappa_1(t) + (x - x_bar) * kappa_2(t) + g(t, x),
        where x_bar is the average age in the data, and sum(g(c)) = 0 and sum(cg(c)) = 0.
        
        Parameters
        ----------
        kappa1_t: numpy.array
            kappa_1 parameters of the M6 model.
        kappa2_t: numpy.array
            kappa_2 parameters of the M6 model.
        gamma_tx: numpy.array
            g parameters of the M6 model.            
            
        Returns
        -------
        dict
            Dictionary of the form 
            {KAPPA1_T: {
                VALUE: kappa1_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, ONES]
                DERIVATIVE: ONES
                VAR: YEAR
                VALUE_NAME: KAPPA1_T
                },
             KAPPA2_T: {
                VALUE: kappa2_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, ONES]
                DERIVATIVE: CENTERED_AGE
                VAR: YEAR
                VALUE_NAME: KAPPA2_T
                },
             GAMMA_TX: {
                VALUE: gamma_tx
                INDEX: g
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, ONES]
                DERIVATIVE: ONES
                VAR: GENERATION
                VALUE_NAME: GAMMA_TX
                }
            }
        """
        A1 = [M6_K1T, M6_K2T, M6_GTX]
        A2 = [COLUMN_ONES, COLUMN_CENTERED_AGE, COLUMN_ONES]
        params = {
            M6_K1T: {
                PARAMS_VALUE: kappa1_t, 
                PARAMS_INDEX: self.t,
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_ONES,
                PARAMS_VARIABLE: COLUMN_YEAR,
                PARAMS_VALUE_NAME: M6_K1T
            },
            M6_K2T: {
                PARAMS_VALUE: kappa2_t, 
                PARAMS_INDEX: self.t,
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_CENTERED_AGE, 
                PARAMS_VARIABLE: COLUMN_YEAR, 
                PARAMS_VALUE_NAME: M6_K2T
            },
            M6_GTX: {
                PARAMS_VALUE: gamma_tx, 
                PARAMS_INDEX: self.g, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_ONES, 
                PARAMS_VARIABLE: COLUMN_GENERATION, 
                PARAMS_VALUE_NAME: M6_GTX
            }
        }
        return params
        
    def constraints(self, params):
        """
        Overides MortalityModel.constraints to adjust for M6 constraints i.e. sum(g(c)) = 0 and sum(cg(c)) = 0.
        
        Parameters
        ----------
        params: dict
            Parameters of the model with the following form 
            {KAPPA1_T: {
                VALUE: kappa1_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, ONES]
                DERIVATIVE: ONES
                VAR: YEAR
                VALUE_NAME: KAPPA1_T
                },
             KAPPA2_T: {
                VALUE: kappa2_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, ONES]
                DERIVATIVE: CENTERED_AGE
                VAR: YEAR
                VALUE_NAME: KAPPA2_T
                },
             GAMMA_TX: {
                VALUE: gamma_tx
                INDEX: g
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, ONES]
                DERIVATIVE: ONES
                VAR: GENERATION
                VALUE_NAME: GAMMA_TX
                }
            }
            
        Returns 
        -------
        dict
            Parameters adjusted to fit the constraints. Same form as the input.
        """
        params_f = params.copy()
        g = np.array(params[M6_GTX][PARAMS_INDEX])
        ones = np.ones(len(params[M6_GTX][PARAMS_INDEX]))
        X = np.array([ones, g]).T
        y = params[M6_GTX][PARAMS_VALUE]
        lr = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
        phi1 = lr[0]
        phi2 = lr[1]
        params[M6_GTX][PARAMS_VALUE] = params[M6_GTX][PARAMS_VALUE] - (phi1 + phi2 * g)
        params[M6_K2T][PARAMS_VALUE] = params[M6_K2T][PARAMS_VALUE] - phi2
        params[M6_K1T][PARAMS_VALUE] = params[M6_K1T][PARAMS_VALUE] + phi1 + phi2 * (self.t - self.data[COLUMN_AGE].mean())
        return params
    
    def fit(self, kappa1_t, kappa2_t, gamma_tx, tol=EPS, max_iter=MAX_ITER):
        """
        Fits M6 model.
        
        Parameters
        ----------
        kappa1_t: numpy.array
            Initial values for kappa1_t.
        kappa2_t: numpy.array
            Initial values for kappa2_t.
        gamma_tx: numpy.array
            Initial values for gamma_tx.
        tol: float
            Error tolerance for the Newton-Raphson algorithm to stop.
        max_iter: int
            Maximum number of iteration of the Newton-Raphson algorithm performed.
        
        Returns
        -------
        numpy.array
            kappa1_t fitted.
        numpy.array
            kappa2_t fitted.
        numpy.array
            gamma_tx fitted.            
        """
        params_init = self.edit_params(kappa1_t, kappa2_t, gamma_tx)
        params_f = self.fit_model(params_init, tol, max_iter)
        kappa1_t_f = params_f[M6_K1T][PARAMS_VALUE]
        kappa2_t_f = params_f[M6_K2T][PARAMS_VALUE]
        gamma_tx_f = params_f[M6_GTX][PARAMS_VALUE]
        return kappa1_t_f, kappa2_t_f, gamma_tx_f

class APC(MortalityModel):
    """
    Class for Age-Period-Cohort model.
    Model has the form:
        eta(x, t) = alpha(x) + kappa_1(t) + g(t, x),
    with sum(kappa_1(t)) = 0, sum(g(c)) = 0 and sum(cg(c)) = 0.
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
        
    def edit_params(self, alpha_x, kappa_t, gamma_tx):
        """
        Create a dictionary from the input to fit a Age-Period-Cohort model:
            eta(x, t) = alpha(x) + kappa_1(t) + g(t, x),
        with sum(kappa_1(t)) = 0, sum(g(c)) = 0 and sum(cg(c)) = 0.
        
        Parameters
        ----------
        alpha_x: numpy.array
            alpha_x parameters of the Age-Period-Cohort model.
        kappa1_t: numpy.array
            kappa_1 parameters of the Age-Period-Cohort model.
        gamma_tx: numpy.array
            g parameters of the Age-Period-Cohort model.            
            
        Returns
        -------
        dict
            Dictionary of the form 
            {ALPHA_X: {
                VALUE: alpha_x
                INDEX: x
                A1: [ALPHA_X, KAPPA1_T, GAMMA_TX]
                A2: [ONES, ONES, ONES]
                DERIVATIVE: ONES
                VAR: AGE
                VALUE_NAME: ALPHA_X
                },
             KAPPA1_T: {
                VALUE: kappa1_t
                INDEX: t
                A1: [ALPHA_X, KAPPA1_T, GAMMA_TX]
                A2: [ONES, ONES, ONES]
                DERIVATIVE: ONES
                VAR: YEAR
                VALUE_NAME: KAPPA1_T
                },
             GAMMA_TX: {
                VALUE: gamma_tx
                INDEX: g
                A1: [ALPHA_X, KAPPA1_T, GAMMA_TX]
                A2: [ONES, ONES, ONES]
                DERIVATIVE: ONES
                VAR: GENERATION
                VALUE_NAME: GAMMA_TX
                }
            }
        """
        A1 = [APC_AX, APC_K1T, APC_GTX]
        A2 = [COLUMN_ONES, COLUMN_ONES, COLUMN_ONES]
        params = {
            APC_AX: {
                PARAMS_VALUE: alpha_x, 
                PARAMS_INDEX: self.x,
                PARAMS_A1: A1,
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_ONES,
                PARAMS_VARIABLE: COLUMN_AGE,
                PARAMS_VALUE_NAME: APC_AX
            },
            APC_K1T: {
                PARAMS_VALUE: kappa_t, 
                PARAMS_INDEX: self.t, 
                PARAMS_A1: A1,
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_ONES,
                PARAMS_VARIABLE: COLUMN_YEAR,
                PARAMS_VALUE_NAME: APC_K1T
            },
            APC_GTX: {
                PARAMS_VALUE: gamma_tx,
                PARAMS_INDEX: self.g, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_ONES, 
                PARAMS_VARIABLE: COLUMN_GENERATION, 
                PARAMS_VALUE_NAME: APC_GTX
            }
        }
        return params
        
    def constraints(self, params):
        """
        Overides MortalityModel.constraints to adjust for Age-Period-Cohort constraints i.e. sum(kappa1(t)) = 0 and sum(g(c)) = 0 and sum(cg(c)) = 0.
        
        Parameters
        ----------
        params: dict
            Parameters of the model with the following form 
            {ALPHA_X: {
                VALUE: alpha_x
                INDEX: x
                A1: [ALPHA_X, KAPPA1_T, GAMMA_TX]
                A2: [ONES, ONES, ONES]
                DERIVATIVE: ONES
                VAR: AGE
                VALUE_NAME: ALPHA_X
                },
             KAPPA1_T: {
                VALUE: kappa1_t
                INDEX: t
                A1: [ALPHA_X, KAPPA1_T, GAMMA_TX]
                A2: [ONES, ONES, ONES]
                DERIVATIVE: ONES
                VAR: YEAR
                VALUE_NAME: KAPPA1_T
                },
             GAMMA_TX: {
                VALUE: gamma_tx
                INDEX: g
                A1: [ALPHA_X, KAPPA1_T, GAMMA_TX]
                A2: [ONES, ONES, ONES]
                DERIVATIVE: ONES
                VAR: GENERATION
                VALUE_NAME: GAMMA_TX
                }
            }
            
        Returns 
        -------
        dict
            Parameters adjusted to fit the constraints. Same form as the input.
        """        
        params_f = params.copy()
        g = np.array(params[APC_GTX][PARAMS_INDEX])
        ones = np.ones(len(params[APC_GTX][PARAMS_INDEX]))
        X = np.array([ones, g]).T
        y = params[APC_GTX][PARAMS_VALUE]
        lr = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
        phi1 = lr[0]
        phi2 = lr[1]
        params[APC_GTX][PARAMS_VALUE] = params[APC_GTX][PARAMS_VALUE] - (phi1 + phi2 * g)
        params[APC_AX][PARAMS_VALUE] = params[APC_AX][PARAMS_VALUE] + phi1 - phi2 * self.x
        params[APC_K1T][PARAMS_VALUE] = params[APC_K1T][PARAMS_VALUE] + phi2 * self.t
        c1 = np.mean(params[APC_K1T][PARAMS_VALUE])
        params[APC_K1T][PARAMS_VALUE] = params[APC_K1T][PARAMS_VALUE] - c1
        params[APC_AX][PARAMS_VALUE] = params[APC_AX][PARAMS_VALUE] + c1
        return params
    
    def fit(self, alpha_x, kappa_t, gamma_tx, tol=EPS, max_iter=MAX_ITER):
        """
        Fits Age-Period-Cohort model.
        
        Parameters
        ----------
        alpha_x: numpy.array
            Initial values for alpha_x.
        kappa1_t: numpy.array
            Initial values for kappa1_t.
        gamma_tx: numpy.array
            Initial values for gamma_tx.
        tol: float
            Error tolerance for the Newton-Raphson algorithm to stop.
        max_iter: int
            Maximum number of iteration of the Newton-Raphson algorithm performed.
        
        Returns
        -------
        numpy.array
            alpha_x fitted.
        numpy.array
            kappa1_t fitted.
        numpy.array
            gamma_tx fitted.            
        """
        params_init = self.edit_params(alpha_x, kappa_t, gamma_tx)
        params_f = self.fit_model(params_init, tol, max_iter)
        alpha_x_f = params_f[APC_AX][PARAMS_VALUE]
        kappa_t_f = params_f[APC_K1T][PARAMS_VALUE]
        gamma_tx_f = params_f[APC_GTX][PARAMS_VALUE]
        return alpha_x_f, kappa_t_f, gamma_tx_f

class M7(MortalityModel):
    """
    Class for M7 model.
    Model has the form:
        eta(x, t) = kappa_1(t) + (x - x_bar) * kappa_2(t) + ((x - x_bar)^2 - sigma_x^2) * kappa_3(t) + g(t, x),
    where x_bar is the average age of the data and sigma_x^2 is the average value of (x - x_bar)^2. 
    Constraints are sum(g(c)) = 0, sum(cg(c)) = 0 and sum(c^2g(c)) = 0.
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
        
    def edit_params(self, kappa1_t, kappa2_t, kappa3_t, gamma_tx):
        """
        Create a dictionary from the input to fit a M7 model:
            eta(x, t) = kappa_1(t) + (x - x_bar) * kappa_2(t) + ((x - x_bar)^2 - sigma_x^2) * kappa_3(t) + g(t, x),
        where x_bar is the average age of the data and sigma_x^2 is the average value of (x - x_bar)^2. 
        Constraints are sum(g(c)) = 0, sum(cg(c)) = 0 and sum(c^2g(c)) = 0.
        
        Parameters
        ----------
        kappa1_t: numpy.array
            kappa_1 parameters of the M7 model.
        kappa2_t: numpy.array
            kappa_2 parameters of the M7 model.
        kappa3_t: numpy.array
            kappa_3 parameters of the M7 model.            
        gamma_tx: numpy.array
            g parameters of the M7 model.            
            
        Returns
        -------
        dict
            Dictionary of the form 
            {KAPPA1_T: {
                VALUE: kappa1_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, KAPPA3_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES]
                DERIVATIVE: ONES
                VAR: YEAR
                VALUE_NAME: KAPPA1_T
                },
             KAPPA2_T: {
                VALUE: kappa2_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, KAPPA3_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES]
                DERIVATIVE: CENTERED_AGE
                VAR: YEAR
                VALUE_NAME: KAPPA2_T
                },
             KAPPA3_T: {
                VALUE: kappa3_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, KAPPA3_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES]
                DERIVATIVE: SIGMA_CENTERED_AGE
                VAR: YEAR
                VALUE_NAME: KAPPA3_T
                },   
             GAMMA_TX: {
                VALUE: gamma_tx
                INDEX: g
                A1: [KAPPA1_T, KAPPA2_T, KAPPA3_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES]
                DERIVATIVE: ONES
                VAR: GENERATION
                VALUE_NAME: GAMMA_TX
                }
            }
        """        
        A1 = [M7_K1T, M7_K2T, M7_K3T, M7_GTX]
        A2 = [COLUMN_ONES, COLUMN_CENTERED_AGE, COLUMN_SIGMA_CENTERED_AGE, COLUMN_ONES]
        params = {
            M7_K1T: {
                PARAMS_VALUE: kappa1_t,
                PARAMS_INDEX: self.t,
                PARAMS_A1: A1,
                PARAMS_A2: A2,
                PARAMS_DERIVATIVE: COLUMN_ONES,
                PARAMS_VARIABLE: COLUMN_YEAR,
                PARAMS_VALUE_NAME: M7_K1T
            },
            M7_K2T: {
                PARAMS_VALUE: kappa2_t,
                PARAMS_INDEX: self.t, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_CENTERED_AGE,
                PARAMS_VARIABLE: COLUMN_YEAR, 
                PARAMS_VALUE_NAME: M7_K2T
            },
            M7_K3T: {
                PARAMS_VALUE: kappa3_t,
                PARAMS_INDEX: self.t, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_SIGMA_CENTERED_AGE,
                PARAMS_VARIABLE: COLUMN_YEAR, 
                PARAMS_VALUE_NAME: M7_K3T
            },
            M7_GTX: {
                PARAMS_VALUE: gamma_tx,
                PARAMS_INDEX: self.g,
                PARAMS_A1: A1, 
                PARAMS_A2: A2,
                PARAMS_DERIVATIVE: COLUMN_ONES,
                PARAMS_VARIABLE: COLUMN_GENERATION, 
                PARAMS_VALUE_NAME: M7_GTX
            }
        }
        return params
        
    def constraints(self, params):
        """
        Overides MortalityModel.constraints to adjust for M7 constraints i.e. sum(g(c)) = 0, sum(cg(c)) = 0 and sum(c^2g(c)) = 0.
        
        Parameters
        ----------
        params: dict
            Parameters of the model with the following form 
            {KAPPA1_T: {
                VALUE: kappa1_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, KAPPA3_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES]
                DERIVATIVE: ONES
                VAR: YEAR
                VALUE_NAME: KAPPA1_T
                },
             KAPPA2_T: {
                VALUE: kappa2_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, KAPPA3_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES]
                DERIVATIVE: CENTERED_AGE
                VAR: YEAR
                VALUE_NAME: KAPPA2_T
                },
             KAPPA3_T: {
                VALUE: kappa3_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, KAPPA3_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES]
                DERIVATIVE: SIGMA_CENTERED_AGE
                VAR: YEAR
                VALUE_NAME: KAPPA3_T
                },   
             GAMMA_TX: {
                VALUE: gamma_tx
                INDEX: g
                A1: [KAPPA1_T, KAPPA2_T, KAPPA3_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, SIGMA_CENTERED_AGE, ONES]
                DERIVATIVE: ONES
                VAR: GENERATION
                VALUE_NAME: GAMMA_TX
                }
            }
            
        Returns 
        -------
        dict
            Parameters adjusted to fit the constraints. Same form as the input.
        """  
        params_f = params.copy()
        x_bar = np.mean(self.x)
        s2 = np.mean((self.x - x_bar)**2)
        g = np.array(params[M7_GTX][PARAMS_INDEX])
        ones = np.ones(len(params[M7_GTX][PARAMS_INDEX]))
        X = np.array([ones, g, g**2]).T
        y = params[M7_GTX][PARAMS_VALUE]
        lr = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
        phi1 = lr[0]
        phi2 = lr[1]
        phi3 = lr[2]
        params[M7_GTX][PARAMS_VALUE] = params[M7_GTX][PARAMS_VALUE] - phi1 - phi2 * g - phi3 * g**2
        params[M7_K3T][PARAMS_VALUE] = params[M7_K3T][PARAMS_VALUE] + phi3
        params[M7_K2T][PARAMS_VALUE] = params[M7_K2T][PARAMS_VALUE] - phi2 - 2 * phi3 * (self.t - x_bar)
        params[M7_K1T][PARAMS_VALUE] = params[M7_K1T][PARAMS_VALUE] + phi1 + phi2 * (self.t - x_bar) + phi3 * ((self.t - x_bar)**2 + s2)
        return params
    
    def fit(self, kappa1_t, kappa2_t, kappa3_t, gamma_tx, tol=EPS, max_iter=MAX_ITER):
        """
        Fits M7 model.
        
        Parameters
        ----------
        kappa1_t: numpy.array
            Initial values for kappa1_t.
        kappa2_t: numpy.array
            Initial values for kappa2_t.
        kappa3_t: numpy.array
            Initial values for kappa3_t.            
        gamma_tx: numpy.array
            Initial values for gamma_tx.
        tol: float
            Error tolerance for the Newton-Raphson algorithm to stop.
        max_iter: int
            Maximum number of iteration of the Newton-Raphson algorithm performed.
        
        Returns
        -------
        numpy.array
            kappa1_t fitted.
        numpy.array
            kappa2_t fitted.
        numpy.array
            kappa3_t fitted.            
        numpy.array
            gamma_tx fitted.            
        """
        params_init = self.edit_params(kappa1_t, kappa2_t, kappa3_t, gamma_tx)
        params_f = self.fit_model(params_init, tol, max_iter)
        kappa1_t_f = params_f[M7_K1T][PARAMS_VALUE]
        kappa2_t_f = params_f[M7_K2T][PARAMS_VALUE]
        kappa3_t_f = params_f[M7_K3T][PARAMS_VALUE]
        gamma_tx_f = params_f[M7_GTX][PARAMS_VALUE]
        return kappa1_t_f, kappa2_t_f, kappa3_t_f, gamma_tx_f
    
class M8(MortalityModel):
    """
    Class for M8 model.
    Model has the form:
        eta(x, t) = kappa_1(t) + (x - x_bar) * kappa_2(t) + (x_c - x) * g(t, x),
    where x_bar is the average age of the data x_c a pivot age. 
    Constraint is sum(g(c)) = 0.
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
    x_c: int
        Pivot age for generation parameters.
    
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
    
    def __init__(self, data, min_year, max_year, min_age, max_age, x_c):
        super().__init__(data, min_year, max_year, min_age, max_age)
        self.x_c = x_c
    
    @property
    def data_calibration(self):
        """
        Function that prepares data for calibration by adding some columns X_C.
        
        Returns
        -------
        pandas.DataFrame
            Mortality data for calibration.
        """        
        data_calibration = super().data_calibration
        data_calibration[M8_XC] = self.x_c - data_calibration[COLUMN_AGE]
        return data_calibration
        
    def edit_params(self, kappa1_t, kappa2_t, gamma_tx):
        """
        Create a dictionary from the input to fit a M8 model:
            eta(x, t) = kappa_1(t) + (x - x_bar) * kappa_2(t) + (x_c - x) * g(t, x),
        where x_bar is the average age of the data x_c a pivot age. 
        Constraint is sum(g(c)) = 0.
        
        Parameters
        ----------
        kappa1_t: numpy.array
            kappa_1 parameters of the M7 model.
        kappa2_t: numpy.array
            kappa_2 parameters of the M7 model.           
        gamma_tx: numpy.array
            g parameters of the M7 model.            
            
        Returns
        -------
        dict
            Dictionary of the form 
            {KAPPA1_T: {
                VALUE: kappa1_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, XC]
                DERIVATIVE: ONES
                VAR: YEAR
                VALUE_NAME: KAPPA1_T
                },
             KAPPA2_T: {
                VALUE: kappa2_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, XC]
                DERIVATIVE: CENTERED_AGE
                VAR: YEAR
                VALUE_NAME: KAPPA2_T
                },  
             GAMMA_TX: {
                VALUE: gamma_tx
                INDEX: g
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, XC]
                DERIVATIVE: XC
                VAR: GENERATION
                VALUE_NAME: GAMMA_TX
                }
            }
        """        
        A1 = [M8_K1T, M8_K2T, M8_GTX]
        A2 = [COLUMN_ONES, COLUMN_CENTERED_AGE, M8_XC]
        params = {
            M8_K1T: {
                PARAMS_VALUE: kappa1_t,
                PARAMS_INDEX: self.t,
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_ONES,
                PARAMS_VARIABLE: COLUMN_YEAR, 
                PARAMS_VALUE_NAME: M8_K1T
            },
            M8_K2T: {
                PARAMS_VALUE: kappa2_t, 
                PARAMS_INDEX: self.t, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: COLUMN_CENTERED_AGE,
                PARAMS_VARIABLE: COLUMN_YEAR, 
                PARAMS_VALUE_NAME: M8_K2T
            },
            M8_GTX: {
                PARAMS_VALUE: gamma_tx, 
                PARAMS_INDEX: self.g, 
                PARAMS_A1: A1, 
                PARAMS_A2: A2, 
                PARAMS_DERIVATIVE: M8_XC,
                PARAMS_VARIABLE: COLUMN_GENERATION,
                PARAMS_VALUE_NAME: M8_GTX
            }
        }
        return params
        
    def constraints(self, params):
        """
        Overides MortalityModel.constraints to adjust for M8 constraints i.e. sum(g(c)) = 0.
        
        Parameters
        ----------
        params: dict
            Parameters of the model with the following form 
            {KAPPA1_T: {
                VALUE: kappa1_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, XC]
                DERIVATIVE: ONES
                VAR: YEAR
                VALUE_NAME: KAPPA1_T
                },
             KAPPA2_T: {
                VALUE: kappa2_t
                INDEX: t
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, XC]
                DERIVATIVE: CENTERED_AGE
                VAR: YEAR
                VALUE_NAME: KAPPA2_T
                },  
             GAMMA_TX: {
                VALUE: gamma_tx
                INDEX: g
                A1: [KAPPA1_T, KAPPA2_T, GAMMA_TX]
                A2: [ONES, CENTERED_AGE, XC]
                DERIVATIVE: XC
                VAR: GENERATION
                VALUE_NAME: GAMMA_TX
                }
            }
            
        Returns 
        -------
        dict
            Parameters adjusted to fit the constraints. Same form as the input.
        """        
        params_f = params.copy()
        x_bar = np.mean(self.x)
        c = np.mean(params[M8_GTX][PARAMS_VALUE])
        params[M8_GTX][PARAMS_VALUE] = params[M8_GTX][PARAMS_VALUE] - c
        params[M8_K1T][PARAMS_VALUE] = params[M8_K1T][PARAMS_VALUE] + c * (self.x_c - x_bar)
        params[M8_K2T][PARAMS_VALUE] = params[M8_K2T][PARAMS_VALUE] - c
        return params
    
    def fit(self, kappa1_t, kappa2_t, gamma_tx, tol=EPS, max_iter=MAX_ITER):
        """
        Fits M8 model.
        
        Parameters
        ----------
        kappa1_t: numpy.array
            Initial values for kappa1_t.
        kappa2_t: numpy.array
            Initial values for kappa2_t.          
        gamma_tx: numpy.array
            Initial values for gamma_tx.
        tol: float
            Error tolerance for the Newton-Raphson algorithm to stop.
        max_iter: int
            Maximum number of iteration of the Newton-Raphson algorithm performed.
        
        Returns
        -------
        numpy.array
            kappa1_t fitted.
        numpy.array
            kappa2_t fitted.            
        numpy.array
            gamma_tx fitted.            
        """        
        params_init = self.edit_params(kappa1_t, kappa2_t, gamma_tx)
        params_f = self.fit_model(params_init, tol, max_iter)
        kappa1_t_f = params_f[M8_K1T][PARAMS_VALUE]
        kappa2_t_f = params_f[M8_K2T][PARAMS_VALUE]
        gamma_tx_f = params_f[M8_GTX][PARAMS_VALUE]
        return kappa1_t_f, kappa2_t_f, gamma_tx_f