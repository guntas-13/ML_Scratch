# **Linear Regression 2-dimensional**
A set of $N$ data points $(x_i, y_i)$, the goal is to find the best linear map $f: \mathbb{R}^2 \to \mathbb{R}^2$ such that $f(x) = mx + b$ fits the data points. In simpler terms, we assume the relation between the dependenet variable $y$ and independent variable $x$ is linear and try finding the optimal $m$ and $b$ such that some error function is minimised. 

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
Arrive at the desired $m$ and $b$ by updating these values following the direction of the greatest descent of this function. The learning rate $L$ has to be specified.
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
![Alt Text]("Linear%20Regression/V1.gif")
