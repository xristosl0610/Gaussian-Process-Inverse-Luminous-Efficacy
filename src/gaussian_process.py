from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from sklearn.utils.optimize import _check_optimize_result


class GPR(GaussianProcessRegressor):
    """
    GPR class for Gaussian Process Regression.

    Args:
        kernel: Kernel object, default=None.
        alpha: float, default=1e-10.
        optimizer: str, default='fmin_l_bfgs_b'.
        n_restarts_optimizer: int, default=0.
        normalize_y: bool, default=False.
        copy_X_train: bool, default=True.
        random_state: int, RandomState instance, default=None.
        max_iter: float, default=2e5.

    Returns:
        None.
    """
    def __init__(self, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                 n_restarts_optimizer=0, normalize_y=False, copy_X_train=True,
                 random_state=None, max_iter=2e5):
        super().__init__(kernel=kernel, alpha=alpha, optimizer=optimizer,
                         n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y,
                         copy_X_train=copy_X_train, random_state=random_state)
        self.max_iter = max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        """
        Conducts constrained optimization using L-BFGS-B method.

        Args:
            obj_func: Objective function to minimize.
            initial_theta: Initial guess for the solution.
            bounds: Bounds for variables.

        Returns:
            Tuple containing the optimized solution x and the minimum value found by the optimization.
        """
        opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True,
                           bounds=bounds, options={'maxiter': self.max_iter})
        _check_optimize_result("lbfgs", opt_res)
        return opt_res.x, opt_res.fun
