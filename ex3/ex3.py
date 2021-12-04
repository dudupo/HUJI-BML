import numpy as np
from matplotlib import pyplot as plt
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior


def cong(H ,u):
    return H.T @ u @ H

def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    sig = model.cov
    n = model.sig

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = model.fit_mu
    map_cov = model.fit_cov

    # calculate the log-evidence
    # <your code here>
    
    diff = map - mu
    psig = np.linalg.pinv( sig)
    w = (y - model.h(X) @ map).T @ (y - model.h(X) @ map) 
    q = len(y) * 2 * np.log(n) 

    return 0.5 * np.linalg.det( model.fit_cov ) / np.linalg.det( sig ) -\
         0.5 * (cong(diff, psig) + n**(-2)*w + q )


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x**2 - 1
    f2 = lambda x: x**3 - 3*x
    f3 = lambda x: x**6 - 15*x**4 + 45*x**2 - 15
    f4 = lambda x: 5*np.exp(3*x) / (1 + np.exp(3*x))
    f5 = lambda x: 2*(np.sin(x*2.5) - np.abs(x))
    functions = [f1, f2, f3, f4, f5]
    x = np.linspace(-3, 3, 500)

    # set up model parameters
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    noise_var = .25
    alpha = 5

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))
        z = []
        # plt.scatter(x, y)
        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            z.append(ev)

        plt.plot(degrees, z)
        plt.show()
        plt.scatter(x,y)
        dmax, dmin = np.argmax(z), np.argmin(z)
        for j, d in enumerate([dmin, dmax]):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            blr = BayesianLinearRegression(mean, cov, noise_var, pbf)
            blr.fit(x,y)

            plt.plot(x , blr.predict(x))
        plt.show()

    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load('nov162020.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        # <your code here>

    # plot log-evidence versus amount of sample noise
    # <your code here>


if __name__ == '__main__':
    main()



