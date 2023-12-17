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
\mathbf{v} = \begin{bmatrix} x_1 \\\ x_2 \\\ \vdots \\\ x_N \end{bmatrix}
```

and 
```math
\nabla F(\mathbf{v}) = \begin{bmatrix} \frac{\partial F}{\partial x_1} \\\ \frac{\partial F}{\partial x_2} \\\ \vdots \\\ \frac{\partial F}{\partial x_n} \end{bmatrix} 
```

Hence, the equation becomes:

```math
\begin{equation}
\begin{bmatrix} x_1^{(i + 1)} \\\ x_2^{(i + 1)} \\\ \vdots \\\ x_N^{(i + 1)} \end{bmatrix} = \begin{bmatrix} x_1^{(i)} \\\ x_2^{(i)} \\\ \vdots \\\ x_N^{(i)} \end{bmatrix} - L \begin{bmatrix}\frac{\partial F}{\partial x_1} \\\ \frac{\partial F}{\partial x_2} \\\ \vdots \\\ \frac{\partial F}{\partial x_n}
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


# **Convolution**
For two functions $f(x)$ and $g(x)$, the convolution function $f(x) * g(x)$ is defined as:

```math
\begin{equation}
(f * g) (t) = \int_{-\infty}^{\infty} f(τ) ⋅ g(t - τ) dτ
\end{equation}
```

for **discrete** samples that we deal with:

```math
\begin{equation}
y[n] = f[n] * g[n] = \sum_{k = -∞}^{∞} f[k] ⋅ g[n - k]
\end{equation}
```

if $f$ has $N$ samples and $g$ has $M$ samples, then the convolved function has $N + M - 1$ samples. A basic rule: **_"flip any one of the functions, overlap it with the stationary one, multiply and add, and then traverse over."_**

# **2D Convolution**
![Alt Text](https://github.com/guntas-13/ML_Scratch/blob/main/V5.gif)

## **Implementing the 2d Convolution**

```math
A = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
```

when zero-padded by 1 pixel gives:
```math
A' = \begin{bmatrix}0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 2 & 3 & 0 \\ 0 & 4 & 5 & 6 & 0 \\ 0 & 7 & 8 & 9 & 0 \\ 0 & 0 & 0 & 0 & 0\end{bmatrix}
```
<br>
This is achieved as:

```python
A_padded = np.pad(A, padding = 1, mode = "constant")
```

Also, before proceeding with the convolution, the kernel must be **flipped Left-Right** and then **Upside-Down** <br>

```math
ker = \begin{bmatrix}a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} ⟶ \begin{bmatrix}c & b & a \\ f & e & d \\ i & h & g \end{bmatrix} ⟶ \begin{bmatrix}i & h & g \\ f & e & d \\ c & b & a \end{bmatrix} = ker'
```
 <br>

This is achieved by:

```python
ker_flipped = np.flipud(np.fliplr(ker))
```

**fliplr** denoting a left-right flip and **flipud** denoting a up-down flip.
Choose a **stride** of length 1 and perform the convolution as the dot product of kernel sized chunks of $A$ with the $ker$:

```math
\begin{bmatrix}0 & 0 & 0 \\ 0 & 1 & 2 \\ 0 & 4 & 5 \end{bmatrix} \cdot \begin{bmatrix}i & h & g \\ f & e & d \\ c & b & a \end{bmatrix} = elt_1
```

 <br><br>

 ```math
\begin{bmatrix}0 & 0 & 0 \\ 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \cdot \begin{bmatrix}i & h & g \\ f & e & d \\ c & b & a \end{bmatrix} = elt_2
```
<br>
.
.
.
<br>

```math
\begin{bmatrix}5 & 6 & 0 \\ 8 & 9 & 0 \\ 0 & 0 & 0 \end{bmatrix} \cdot \begin{bmatrix}i & h & g \\ f & e & d \\ c & b & a \end{bmatrix} = elt_N
```

 <br><br>
Notice the dimensions of the final output matrix:

```math
\begin{equation}
R_{\text{height}} = \frac{A_{\text{height}} + 2\cdot\text{padding} - ker_{\text{height}}}{\text{stride}} + 1
\end{equation}
```

```math
\begin{equation}
R_{\text{width}} = \frac{A_{\text{width}} + 2\cdot\text{padding} - ker_{\text{width}}}{\text{stride}} + 1
\end{equation}
```

## **Function for the convolution**
```python
def convolve2d(image, kernel, padding, stride):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (image_height + 2 * padding - kernel_height) // stride + 1
    output_width = (image_width + 2 * padding - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width))

    padded_image = np.pad(image, padding, mode = "constant")
    kernel = np.flipud(np.fliplr(kernel))

    for i in range(0, output_height, stride):
        for j in range(0, output_width, stride):
            output[i, j] = np.sum(padded_image[i : i + kernel_height, j : j+kernel_width] * kernel)

    return output
```

## **The Edge Detection using Kernels - _Sobel Operators_**

```math
\mathbf{G_x} = \begin{bmatrix}1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix}
```

```math
\mathbf{G_y} = \begin{bmatrix}1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix}
```

Obtain two images $A_x$ and $A_y$ for detecting **vertical** and **horizontal** edges as:

```math
\mathbf{A_x} = \mathbf{G_x} * \mathbf{A} = \begin{bmatrix}1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix} * \mathbf{A}
```

```math
\mathbf{A_y} = \mathbf{G_y} * \mathbf{A} = \begin{bmatrix}1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix} * \mathbf{A}
```

The final **edge-detected** image is obtained as:

```math
\mathbf{A_{\text{sobel}}} = \sqrt{\mathbf{A_x}^2 + \mathbf{A_y}^2}
```

## **Original Image $(3264, 4928)$**
![Alt Text](https://github.com/guntas-13/ML_Scratch/blob/main/Image.jpeg)

## **Sobel Image $(3264, 4928)$**
![Alt Text](https://github.com/guntas-13/ML_Scratch/blob/main/Sobel.jpeg)

### **$A_x$ and $A_y$**
![Alt Text](https://github.com/guntas-13/ML_Scratch/blob/main/GradX_GradY.png)

### **Orginal - Grayscale - Sobel**
![Alt Text](https://github.com/guntas-13/ML_Scratch/blob/main/EdgeDetect.png)
