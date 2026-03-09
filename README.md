**Linear Regression with Regularization - Guide**

This README summarizes the essential theory, formulas, code equivalents, and practical tips to understand and implement linear regression with regularization (mainly L2 / Ridge). It's intended to be a compact yet practical reference for both scratch implementations and common ML libraries (e.g., PyTorch).

**Overview**
- **Goal:** Predict a scalar target y from features x using a linear model. Regularization penalizes large weights to reduce overfitting.
- **Model (hypothesis):** $h_{\theta}(x) = \theta^T x = w^T x + b$ where $\theta = [b, w_1, \dots, w_n]$ or equivalently treat bias separately.

**Cost / Loss (with L2 regularization)**
- Mean squared error (with L2 penalty on weights, not bias):

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

- Notes:
  - $m$ is number of examples.
  - Regularization coefficient $\lambda \ge 0$. Larger $\lambda$ increases penalty on weights (shrinks them).
  - The bias term $b$ (or $\theta_0$) is typically NOT regularized.

**Gradient (component-wise)**
- For weight $w_j$ (j ≥ 1):

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j
$$

- For bias $b$ (or $\theta_0$):

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})
$$

**Vectorized form**
- Let X be $m \times (n+1)$ design matrix with a column of ones for bias, or use $m\times n$ and handle bias separately.

Hypothesis: $\mathbf{h} = Xw + b$ (if bias separate) or $\mathbf{h} = X\theta$ (if included).

Cost: $J = \frac{1}{2m} ||X\theta - y||^2 + \frac{\lambda}{2m} ||w||^2$ (exclude bias from the regularizer).

Gradient (vectorized, with bias separate):

$$
\nabla_w J = \frac{1}{m} X^T (Xw + b - y) + \frac{\lambda}{m} w
$$

$$
\nabla_b J = \frac{1}{m} \sum (Xw + b - y)
$$

**Optimization approaches**
- Analytical (normal equation with L2):

If you include regularization and center/scale features appropriately, closed form solution for Ridge is:

$$
\theta = (X^T X + \lambda I)^{-1} X^T y
$$

Note: don't regularize the bias - replace the top-left element accordingly or center inputs.

- Gradient descent / batch gradient descent: use the gradients above and update: $w := w - \alpha \nabla_w J$ and $b := b - \alpha \nabla_b J$.

**Scratch Python (vectorized NumPy) - equivalent code snippet**

```python
import numpy as np

def compute_cost(X, y, w, b, lam):
    m = X.shape[0]
    preds = X.dot(w) + b
    err = preds - y
    cost = (err**2).sum() / (2*m)
    reg = (lam / (2*m)) * (w**2).sum()
    return cost + reg

def gradient_step(X, y, w, b, lam, lr):
    m = X.shape[0]
    preds = X.dot(w) + b
    err = preds - y
    grad_w = (X.T.dot(err)) / m + (lam / m) * w
    grad_b = err.sum() / m
    w = w - lr * grad_w
    b = b - lr * grad_b
    return w, b

# training loop
# w = np.zeros(n_features); b = 0.0
# for it in range(iters): w, b = gradient_step(X, y, w, b, lam, lr)
```

**PyTorch equivalent (L2 via optimizer weight decay)**

```python
import torch
import torch.nn as nn

model = nn.Linear(n_features, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=lambda_val)
# weight_decay in optimizers implements L2 penalty on ALL parameters (including bias).
# To exclude bias from regularization, set parameter groups:
params = [
    {'params': [p for n,p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0},
    {'params': [p for n,p in model.named_parameters() if 'weight' in n], 'weight_decay': lambda_val}
]
optimizer = torch.optim.SGD(params, lr=0.01)

# training step
# preds = model(X)
# loss = nn.MSELoss()(preds, y)
# loss.backward(); optimizer.step(); optimizer.zero_grad()
```

**How PyTorch L2 relates to formula**
- `weight_decay` multiplies parameter values during optimizer step and is equivalent to adding $\frac{\lambda}{2m} ||w||^2$ to the loss (scaling differs by optimizer step size and library conventions; treat `weight_decay` as the effective regularization hyperparameter to tune).

**Tips & Tricks**
- Feature scaling: Always standardize/scale continuous features (zero mean, unit variance) before regularization - regularizer is sensitive to feature scale.
- Centering: If you center features (mean=0) you can regularize all coefficients including intercept more safely.
- Choose $\lambda$ by cross-validation: do a grid search on logarithmic scale (e.g., 1e-4 … 1e2).
- Start with small $\lambda$ and increase until validation error stops decreasing.
- Learning rate: tune learning rate and learning schedule; too large will diverge.
- Exclude bias from regularization (recommended) because bias simply shifts predictions and penalizing it can worsen fit.
- Polynomial features: when adding polynomial or interaction terms, regularization helps control explosion of coefficients.
- Regularization effect: increases bias, reduces variance - reduces overfitting at cost of underfitting if too large.
- L1 vs L2: L1 (Lasso) promotes sparsity (feature selection). L2 (Ridge) shrinks weights smoothly.
- Elastic Net: combine L1+L2 when you want both sparsity and stability.

**Debugging & Understanding**
- If training loss decreases but validation loss increases: you're overfitting -> increase $\lambda$ or simplify model.
- If both training and validation loss are high: your model underfits -> decrease $\lambda$ or add features/polynomial terms.
- Inspect coefficient magnitudes: very large coefficients often indicate feature scale problems or lack of regularization.
- Check design matrix for collinearity - Ridge helps with multicollinearity; LASSO may arbitrarily drop correlated features.

**Common pitfalls**
- Regularizing standardized vs raw features: regularize after scaling; otherwise coefficients reflect scale, not importance.
- Forgetting to exclude bias from regularization when using closed-form equations or implementing gradients.
- Misinterpreting `weight_decay` magnitude in libraries - tune it like any hyperparameter.

**Practical workflow**
1. Clean and preprocess data: handle missing values, encode categorical variables, and scale numeric features.
2. Start with a baseline linear regression (no regularization) and evaluate train/val errors.
3. Add L2 regularization (Ridge) and search for best $\lambda$ via cross-validation.
4. Compare performance to L1 (Lasso) and ElasticNet if feature selection or sparsity is desired.
5. If non-linearities exist, consider polynomial or interaction features and re-run regularization tuning.

**Further reading**
- Andrew Ng - CS229 / Coursera notes on regularization


---
