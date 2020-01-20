# Analyzing Objective Functions and Optimization Methods in Classification Task

In this project I implemented various optimization methods and loss functions for classification tasks. By applying the different loss functions and optimization techniques across different datasets, I gained an understanding of the efficiency vs. accuracy tradeoff of each model, and under which conditions they thrive.

### Loss Functions:
**1) Hinge-loss**
$$\mbox{minimize}_{x\in\mathbb{R}^d, \beta \in \mathbb{R}} \ \frac{1}{n} \sum_{i=1}^n \max \{0,1-b_i(a_i^Tx + \beta)\},$$
Where $a_i\in\mathbb{R}^d$ is the feature vector for sample $i$ and $b_i$ is the label of sample $i$. Note that this function is non-smooth.

**2) Hinge-loss Smooth Approximation**

$$
\psi_\mu(z) = 
\begin{cases}
0 & z\ge 1\\
(1-z)^2 & \mu < z < 1 \\
(1-\mu)^2 + 2(1-\mu)(\mu-z) & z \le \mu.
\end{cases}
$$


**3) L2-regularized logistic regression**
$$\mbox{minimize}_{x\in\mathbb{R}^d,\beta\in\mathbb{R}} \ \lambda \|x\|_2^2 + \frac{1}{n} \sum_{i=1}^n \log (1+ \exp(-b_i(a_i^Tx + \beta))).$$
### Optimization Methods:
i) Stochastic sub-gradient

ii) Stochastic gradient

iii) Mini-batch (sub-)gradient

iv) Stochastic average sub-gradient (SAG)

v) Stochastic average gradient (SAG)

vi) Gradient descent with Armijo line-search

vii) Acceleratd gradient with Armijo line-search

### Objective Functions

