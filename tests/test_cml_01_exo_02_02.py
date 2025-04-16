import pytest
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from assignment import create_dataset, mse_loss_jax, infer_jax, regress_loss_jax, gd_regression_jax

def test_loss():
    # Basic tests
    assert mse_loss_jax(np.ones(10), np.ones(10)) == 0     # All right
    assert mse_loss_jax(np.zeros(10), np.ones(10)) == 1.0    # All wrong
    # Compare the implementation to a reference one (scikit-learn)
    for i in range(10):
        y_true = np.random.randn(100)
        y_bar = np.random.randn(100)
        assert (mse_loss_jax(y_true, y_bar) - mean_squared_error(y_true, y_bar)) < 1e-5

def test_regression(reference_test=False):
    # Basic tests
    w0, w1, loss_history = gd_regression_jax(np.random.rand(150), np.random.rand(150), 1000, 1e-2, 2, regress_loss_jax)
    assert len(loss_history) == 1000    # Filled all iterations
    n_iters = 200
    train_split = 400
    # Here simply check if we predict correctly
    for i in range(10):
        degree = np.random.randint(3) + 2
        # Simple (noise-free) classification# degree + 1 for w_0 
        params = np.random.randint(-6, 6, size = degree)
        x_data = np.linspace(-1, 1, 100)
        poly = np.poly1d(params)
        y = poly(x_data)
        w0, w1, loss_history = gd_regression_jax(x_data[:train_split], y[:train_split], n_iters, 1e-2, degree, regress_loss_jax)
        # Compare predictions
        np_pred = infer_jax(x_data[train_split:], w0, w1, degree)
        assert sum(y[train_split:] - np_pred) < np.finfo(np.float32).eps
        # Helper code : Reference (scikit-learn)
        if (reference_test):
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(x.reshape(-1, 1))
            poly_reg_model = LinearRegression().fit(poly_features, y)
            y_predicted = poly_reg_model.predict(poly_features)