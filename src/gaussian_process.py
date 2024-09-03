import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from sklearn.utils.optimize import _check_optimize_result
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import gpy


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


class UncertainGP(GaussianProcessRegressor):
    """
    A Gaussian Process regressor that can handle noisy inputs.
    """

    def __init__(self):
        super().__init__()


class UncertainSGPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            kernel: GPy.kern.Kern | None = None,
            inference: str = "vfe",
            X_variance: float | np.ndarray | None = None,
            n_inducing: int = 10,
            max_iters: int = 200,
            optimizer: str = "scg",
            n_restarts: int = 10,
            verbose: int | None = None,
            normalize_y: bool = False,
            batch_size: int | None = None,
    ):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.X_variance = X_variance
        self.inference = inference
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.verbose = verbose
        self.normalize_y = normalize_y
        self.batch_size = batch_size

    def fit(self, X, y):
        # Check inputs
        X, y = check_X_y(
            X, y, multi_output=True, y_numeric=True, ensure_2d=True, dtype="numeric"
        )

        # normalize outputs
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = np.std(y, axis=0)

            # remove mean to make unit variance
            y = (y - self._y_train_mean) / self._y_train_std
        n_samples, d_dimensions = X.shape

        # default Kernel Function
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

        # Get inducing points
        z = kmeans2(X, self.n_inducing, minit="points")[0]

        # Get Variance
        X_variance = self._check_X_variance(self.X_variance, X.shape)

        # Inference function
        if self.inference.lower() == "vfe" or X_variance is not None:
            inference_method = GPy.inference.latent_function_inference.VarDTC()

        elif self.inference.lower() == "fitc":
            inference_method = GPy.inference.latent_function_inference.FITC()

        else:
            raise ValueError(f"Unrecognized inference method: {self.inference}")

        # Kernel matrix

        if self.batch_size is None:
            gp_model = GPy.models.SparseGPRegression(
                X, y, kernel=self.kernel, Z=z, X_variance=X_variance
            )
        else:
            gp_model = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
                Y=y,
                X=X,
                input_dim=X.shape,
                kernel=self.kernel,
                Z=z,
                X_variance=X_variance,
                inference_method=inference_method,
                batchsize=self.batch_size,
                likelihood=GPy.likelihoods.Gaussian(),
                stochastic=False,
                missing_data=False,
            )

        # set the fitc inference

        # Optimize
        gp_model.optimize(
            self.optimizer, messages=self.verbose, max_iters=self.max_iters
        )

        # Make likelihood variance low to start
        gp_model.Gaussian_noise.variance = 0.01

        # Optimization
        if self.n_restarts >= 1:
            gp_model.optimize_restarts(
                num_restarts=self.n_restarts,
                robust=True,
                verbose=self.verbose,
                max_iters=self.max_iters,
            )
        else:
            gp_model.optimize(
                self.optimizer, messages=self.verbose, max_iters=self.max_iters
            )

        self.gp_model = gp_model

        return self

    def _check_X_variance(
            self, X_variance:  float | np.ndarray | None, X_shape: tuple[int, int]
    ) -> np.ndarray | None:
        """Private method to check the X_variance parameter

        Parameters
        ----------
        X_variance : float, None, np.ndarray 
            The input for the uncertain inputs

        Returns
        -------
        X_variance : np.ndarray, (n_features, n_features)
            The final matrix for the uncertain inputs.
        """
        if X_variance is None:
            return X_variance

        elif isinstance(X_variance, float):
            return X_variance * np.ones(shape=X_shape)

        elif isinstance(X_variance, np.ndarray):
            if X_variance.shape == 1:
                return X_variance * np.ones(shape=X_shape)
            elif X_variance.shape[0] == X_shape[1]:
                return np.tile(self.X_variance, (X_shape[0], 1))
            elif X_variance.shape == (X_shape[0], X_shape[1]):
                return X_variance
            else:
                raise ValueError(
                    f"Shape of 'X_variance' {X_variance.shape} "
                    f"doesn't match X {X_shape}"
                )
        else:
            raise ValueError("Unrecognized type of X_variance.")

    def display_model(self):
        return self.gp_model

    def predict(
            self, X, return_std=False, full_cov=False, noiseless=True,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = check_array(X, ensure_2d=True, dtype="numeric")

        include_likelihood = not noiseless
        mean, var = self.gp_model.predict(X, include_likelihood=include_likelihood)

        # undo normalization
        if self.normalize_y:
            mean = self._y_train_std * mean + self._y_train_mean
            var = var * self._y_train_std ** 2

        return (mean, np.sqrt(var)) if return_std else mean

    def _variance_correction(self, X: np.ndarray) -> np.ndarray:
        x_der, _ = self.gp_model.predictive_gradients(X)

        return x_der[..., 0] @ self.X_variance @ x_der[..., 0].T
