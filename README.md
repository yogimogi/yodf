## A 'Hello, World!' forward mode autodiff library.
This library with around 500 lines of code is meant as an illustration of how forward mode autodiff can possibly be implemented. 
It lets you compute the value and the derivative of a function where function is expressed as a computational flow using the primitives
provided by the library. Interface of the library is very similar to Tensorflow 1.15. All the samples provided in *examples* folder 
can very well be run if you do **import tensorflow as tf** as opposed to **import yodf as tf** 

### Installation
**pip install yodf** will install the library. Only dependency it has is *numpy*.
Samples provided in examples folder also have dependency on *matplotlib*.

### Basic usage

Below code computes the value and the derivative of the function **x^2** as x=5.0  
```
import yodf as tf
x = tf.Variable(5.0)
cost = x**2
with tf.Session() as s:
    # global_variables_initializer API added just so as to
	# resemble Tensorflow, it hardly does anything
    s.run(tf.global_variables_initializer())
    s.run(cost)
print(x.value, cost.value, cost.gradient)
```

### Basic gradient descent example
Below code computes optima of the function **x^2** along with the value at which optima occurs starting with x=5.0  
```
import yodf as tf
x = tf.Variable(5.0)
cost = x**2
train = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(cost)
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    for _ in range(50):
        _, cost_final, x_final = s.run([train, x, cost])
print(f"Minima: {cost_final:.10f} x at minima: {x_final:.10f}")
```

## How does it work?
It has a class called *Tensor* with *Variable* and *_Constant* as classes derived from it. Tensor has a value and a gradient.
Gradient of a constant is 0 and that of a variable is 1 which is as good as saying d(x)/dx.  
A tensor can also represent an operation and a tensor representating an operation gets created using a convenient function call.
```
import numpy as np
import yodf as tf
x = tf.Variable(np.array([[1,1],[2,2]]))
op_sin = tf.sin(x)
print(op_sin)
```
Would print **<yod.Tensor type=TensorType.INT, shape=(2, 2), operation='sin'>**  
You typically pass a tensor to run method of *Session* class which ends up evaluating the tensor along with its derivative.
Execute method of tensor just knows how to compute derivative of basic arithmatic operations, power function and some of the 
transcendental functions like sin, cos, log, exp. It also knows how to compute derivative when matrix multiplication operation is 
involved. By applying the chain rule repeatedly to these operations, derivative of an arbitrary function 
(represented as a tensor) gets computed automatically. *run* method simply builds post order traversal tree of the tensor passed to it and evaluates all the nodes in
the tree. *GradientDescentOptimizer* simply updates the value of the variable based on the gradient of the cost tensor passed to 
its minimize function.  
When there are multiple independent variables whose partial derivates needs to be computed, gradient of all but one variable 
whose partial derivative is being computed are set to 0 during computational flow path. This is handled by *GradientDescentOptimizer*.

## Limitiation of forward mode autodiff
Though with forward mode autodiff, derivative of a function with one independent variables gets computed during forward pass itself and 
no backward pass is needed as is the case with reverse mode autodiff (generalized backpropagation), with multiple indepdent variables 
(say weights in a neural network), as many passes are needed as number of indepdent variables. So as can be seen in sample 
https://github.com/yogimogi/yodf/blob/master/examples/example3_linear_regression.ipynb, time needed by gradient descent linearly
increases with increase in degree of polynomial you are trying to fit.  
Execution times will become prohibitively high when trying to fit 
a model with large number of weights which is typically case with deep neural networks.