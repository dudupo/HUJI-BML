import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        # <your code here>
        return np.vander(x, degree) 
    return pbf

def gen_basis_functions(x, func):
    def _execute( _func ):
        return _func(x)
    funcs = np.vectorize(func)          
    return np.vectorize(_execute)(funcs)

def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Create a function that calculates Gaussian basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :param beta: a float depicting the lengthscale of the Gaussians
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Gaussian basis functions, a numpy array of shape [N, len(centers)+1]
    """
    def gaussian( center ):
        def _gaussian(z):
            return  np.exp( -( (z-center)**2 ) / (2*(beta**2)) ) / ( np.sqrt(2 * np.pi) * beta)
        return _gaussian
    gaussians = np.vectorize(gaussian)(centers)
    
    def gbf(x: np.ndarray):
        ret = np.zeros( shape=(len(centers), len(x)) )
        for _,gaus in enumerate(gaussians):
            ret[_] = gaus(x)
        return ret.T
    return gbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        def knot(z, initpoint):
            return  np.max( np.array([np.zeros(len(z)) , (z-initpoint)**3 ]) , axis=0)
        poly3_matrix = np.vander(x, 4).transpose()
        ret = np.ones(shape=( len(knots), len(x) ))
        
        for _, initpoint in enumerate(ret):
            ret[_] = knot(x, np.ones(len(x)) *initpoint)
        ret = np.concatenate((poly3_matrix, ret))
        return ret.T
    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        # todo <your code here>
        thetas.append(ln.w)  # append learned parameters here

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    print(cov.shape)
    
    return mu, cov

def conj(H, x):
    return H.T @ x @ H

class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.theta_mean = theta_mean
        self.theta_cov = theta_cov
        self.sig = sig
        self.basis_functions = basis_functions
        self.mu = None
        # <your code here>
        # return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        design = self.basis_functions(X)
        self.stablecalc = np.linalg.inv(self.theta_cov) + design.T @ design/self.sig**2 
        self.cov = np.linalg.inv(self.stablecalc) 
        
        self.mu = np.linalg.solve( self.stablecalc, \
            np.linalg.solve( self.theta_cov, self.theta_mean) + design.T @ y/self.sig**2  )
        
        print(f"mu: {self.mu}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        # <your code here>
        return self.basis_functions(X) @ self.mu

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        # <your code here>
        return np.sqrt( conj(self.basis_functions(X), self.cov ) + np.ones(len(X))*(self.sig**2)) 

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        # <your code here>
        return self.basis_functions(X) @ self.mu


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.basis_functions = basis_functions
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        design = self.basis_functions(X)
        

        # print(design.T @ design)
        Z = np.linalg.pinv( design.T @ design  )
        self.w = Z @ design.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        return  self.basis_functions(X) @ self.w
        

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def plot_func(data, train, model, _filename):
    # print(data.shape, train.shape)
    print(train)
    plt.scatter(data, train, c='b', s=1)
    inp = np.linspace( min(data), max(data), 100 )
    plt.plot(inp, model.predict(inp))
    plt.savefig(f"{_filename}.svg")
    plt.clf()

def main():
    # load the data for November 16 2020
    nov16 = np.load('nov162020.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16)//2]
    train_hours = nov16_hours[:len(nov16)//2]
    test = nov16[len(nov16)//2:]
    test_hours = nov16_hours[len(nov16)//2:]

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)

        # print average squared error performance
        print(f'Average squared error with LR and d={d} is {np.mean((test - ln.predict(test_hours))**2):.2f}')
        # plot graphs for linear regression part
        plot_func(train_hours, train, ln, f"{d}-linear-req" )
    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees
    beta = 2.5  # lengthscale for Gaussian basis functions

    # sets of centers S_1, S_2, and S_3
    centers = [np.array([6, 12, 18]),
               np.array([4, 8, 12, 16, 20]),
               np.array([3, 6, 9, 12, 15, 18, 21])]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]


    def plot_hours_bay(pbf, mu, blr, _str):
        
        for t in temps:
            plt.scatter(  hours, t, c='b', s=1)
        ln = LinearRegression(pbf)
        ln.w = mu
        plot_func( hours, temps[0] , ln , f"{_str}-linear-req-s" )
        blr.fit(np.repeat(hours, len(temps)), temps.flatten())
        plot_func( hours, temps[0] , blr, f"{_str}-baylinear-req-s" )
        
        # plt.fill_between(x, mean-std, mean+std, alpha=.5, label='confidence interval')

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        blr = BayesianLinearRegression(mu, cov, sigma, pbf)
        
        # for t in temps:
        #     plt.scatter(  hours, t, c='b', s=1)
        
        # ln = LinearRegression(pbf)
        # ln.w = mu
        # plot_func( hours, temps[0] , ln , f"{deg}-linear-req-1" )
        # blr.fit(np.repeat(hours, len(temps), temps.flatten()))
        # plot_func( hours, temps[0] , blr, f"{deg}-baylinear-req-1" )

        plot_hours_bay(pbf, mu, blr, f"{deg}")

        # plot prior graphs
        # <your code here>

        # plot posterior graphs
        # <your code here>

    # ---------------------- Gaussian basis functions
    for ind, c in enumerate(centers):
        rbf = gaussian_basis_functions(c, beta)
        mu, cov = learn_prior(hours, temps, rbf)

        blr = BayesianLinearRegression(mu, cov, sigma, rbf)
        # for t in temps:
            # plt.scatter(  hours, t, c='b', s=1)
        # plot_func(hours, temps[0], blr, f"{ind}-baylinear-req-2" )

        plot_hours_bay(rbf, mu, blr, f"{ind}-gaus")

        # plot prior graphs
        # <your code here>

        # plot posterior graphs
        # <your code here>

    # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        blr = BayesianLinearRegression(mu, cov, sigma, spline)
        for t in temps:
            plt.scatter(  hours, t, c='b', s=1)
        # plot_func(hours, temps[0], blr, f"{ind}-baylinear-req-3" )
        plot_hours_bay(spline, mu, blr, f"{ind}-spline")

        # plot prior graphs
        # <your code here>

        # plot posterior graphs
        # <your code here>


if __name__ == '__main__':
    main()
