import pytest
import numpy as np
from sklearn.metrics import hinge_loss
from sklearn.linear_model import LogisticRegression
from assignment import create_dataset, hinge_loss_np, classify_np, gd_classification_np

def test_loss():
    # Basic tests
    assert hinge_loss_np(np.ones(10), np.ones(10)) == 0     # All right
    assert hinge_loss_np(-np.ones(10), np.ones(10)) == 2.0    # All wrong
    assert hinge_loss_np(np.array([1, 1, 1, -1]), np.array([1, 1, -1, -1])) == 0.5    # 1/4 wrong
    assert hinge_loss_np(np.array([1, -1, 1, -1]), np.array([1, 1, -1, -1])) == 1.0    # 1/2 wrong
    # Compare the implementation to a reference one (scikit-learn)
    for i in range(10):
        y_true = (np.random.randint(2, size=100) * 2) - 1
        y_bar = (np.random.randint(2, size=100) * 2) - 1
        assert hinge_loss_np(y_true, y_bar) == hinge_loss(y_true, y_bar)

def test_classification():
    # Basic tests
    x_data, y_classes = create_dataset(200, 0.0, [-2, -1], [2, 1])
    w0, w1, w2, loss_history = gd_classification_np(x_data[:150], y_classes[:150], 1000, 1e-2, hinge_loss_np)
    assert len(loss_history) == 1000    # Filled all iterations
    # Compare the implementation to a reference one (scikit-learn)
    n_iters = 200
    train_split = 400
    for i in range(10):
        c = np.random.randint(5, size=2) + 1
        # Simple (noise-free) classification
        x_data, y_classes = create_dataset(500, 0.0, [-2, -1], [2, 1])
        w0, w1, w2, loss_history = gd_classification_np(x_data[:train_split], y_classes[:train_split], n_iters, 1e-2, hinge_loss_np)
        # Reference (scikit-learn)
        clf = LogisticRegression(random_state=0).fit(x_data[:train_split], y_classes[:train_split])
        # Compare predictions
        clf_pred = (clf.predict(x_data[train_split:]))
        np_pred = classify_np(x_data[train_split:], w0, w1, w2)
        assert sum(clf_pred - np_pred) == 0