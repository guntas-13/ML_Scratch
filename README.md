# **Linear Regression 2-dimensional**
A set of $N$ data points $(x_i, y_i)$, the goal is to find the best linear map $f: \mathbb{R}^2 \to \mathbb{R}^2$ such that $f(x) = mx + b$ fits the data points. In simpler terms, we assume the relation between the dependent variable $y$ and independent variable $x$ is linear and try finding the optimal $m$ and $b$ such that some error function is minimized. 

## **Loss/Error Function**
```math
\begin{equation}
E = \frac{1}{N}\sum_{i = 1}^{N}(y_i - \widehat{y})^2
\end{equation}
```
where,
$\widehat{y} = mx_i + b$, hence
```math
\begin{equation}
E = \frac{1}{N}\sum_{i = 1}^{N}(y_i - (mx_i + b))^2
\end{equation}
```


## **Optimal $m$ and $b$**
```math
\begin{equation}
\frac{∂E}{∂m} = -\frac{2}{N}\sum_{i = 1}^{N}(x_i \times (y_i - (mx_i + b)) )
\end{equation}
```

```math
\begin{equation}
\frac{∂E}{∂b} = -\frac{2}{N}\sum_{i = 1}^{N}(y_i - (mx_i + b))
\end{equation}
```
## **Gradient Descent**
Arrive at the desired $m$ and $b$ by updating these values following the direction of the greatest descent of this function. The ***learning rate*** $L$ has to be specified.
```math
\begin{equation}
\bar{m} = m - L\frac{∂E}{∂m}
\end{equation}
```

```math
\begin{equation}
\bar{b} = b - L\frac{∂E}{∂b}
\end{equation}
```

## **Visualizing the Loss Function $E(m, b)$**
![Alt Text](https://github.com/guntas-13/ML_Scratch/blob/main/V1.gif)

## **Visualizing the Gradient Descent every 5 epochs**
Since $E(m, b)$ has only a single global minima (A Paraboloid!), setting an appropriate **learning rate** $L$ is highly important. Too large a rate will eventually shoot off and skip the minima, which may or may not wobble down to the well upon even increasing the epochs.
![Alt Text](https://github.com/guntas-13/ML_Scratch/blob/main/V2.gif)


# **Gradient Descent**
For a multivariable function, in $N$ dimensions let's say, $F(\textbf{v})$ which is differentiable at a point $\mathbf{v}$, we say that $F(\mathbf{v})$ decreases fastest in the direction of negative of the gradient at that point $\mathbf{v}$ denoted by $-∇F(\mathbf{v})$.

## **Evaluating the descent**
```math
\begin{equation}
\mathbf{v}_{i + 1} = \mathbf{v}_{i} - L \nabla F(\mathbf{v}_{i})
\end{equation}
```

where, $L$ is the learning rate and $L \in \mathbb{R}_{+}$ and 

```math
$\mathbf{v} = \begin{bmatrix} x_1 \\\ x_2 \\\ \vdots \\\ x_N \end{bmatrix} $
```

and 
```math
\nabla F(\mathbf{v}) = \begin{bmatrix} \frac{\partial F}{\partial x_1} \\\ \frac{\partial F}{\partial x_2} \\\ \vdots \\\ \frac{\partial F}{\partial x_n} \end{bmatrix} 
```

Hence, the equation becomes:

```math
\begin{equation}
\begin{bmatrix} x_1^{i + 1} \\\ x_2^{i + 1} \\\ \vdots \\\ x_N^{i + 1} \end{bmatrix} = \begin{bmatrix} x_1^i \\\ x_2^i \\\ \vdots \\\ x_N^i \end{bmatrix} - L \begin{bmatrix}\frac{\partial F}{\partial x_1} \\\ \frac{\partial F}{\partial x_2} \\\ \vdots \\\ \frac{\partial F}{\partial x_n}
\end{bmatrix}
\end{equation}
```

## **Descent to _a minima_**
Notice that the decrease in $F(\mathbf{v})$ is guaranteed only to the nearest well, which may or may not be the global minima. We may run for a specified number of epochs or terminate at a set tolerance too.

## **2-dimensional example**
Let

```math
F(\mathbf{v}) = F\left(\begin{bmatrix} x \\\ y \end{bmatrix}\right) = \sin(x)\cos(y)
```

then the gradient 

```math
∇F(\mathbf{v}) = \begin{bmatrix}\frac{\partial F}{\partial x} \\\ \frac{\partial F}{\partial y} \end{bmatrix} = \begin{bmatrix} \cos(x)\cos(y) \\\ -\sin(x)\sin(y) \end{bmatrix}
```

and starting from an initial point $\mathbf{v}_0$, we may reach the nearest local minima as:

```math
\begin{equation}
\begin{bmatrix} \bar x \\\ \bar y \end{bmatrix} = \begin{bmatrix} x \\\ y \end{bmatrix} - L \begin{bmatrix} \cos(x)\cos(y) \\\ -\sin(x)\sin(y) \end{bmatrix}
\end{equation}
```

## **$F(x, y) = \sin(x)\cos(y)$**
![Alt Text](https://github.com/guntas-13/ML_Scratch/blob/main/V3.gif)

## **Visualizing the Descent**
![Alt Text](https://github.com/guntas-13/ML_Scratch/blob/main/V4.gif)
