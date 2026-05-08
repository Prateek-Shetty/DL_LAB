import tensorflow as tf
import numpy as np

# 1. Tensor Creation
tensor = tf.constant([100, 200, 300])
print(tensor)
print(tensor.shape)
print(tensor.dtype)

# 2. Element-wise Addition
a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[6, 5, 4], [3, 2, 1]])

add = tf.add(a, b)
print(add.numpy())

# 3. Element-wise Subtraction
sub = tf.subtract(a, b)
print(sub.numpy())

# 4. Element-wise Multiplication
mul = tf.multiply(a, b)
print(mul.numpy())

# 5. Element-wise Division
x = tf.constant([10, 20, 30], dtype=tf.float32)
y = tf.constant([2, 4, 5], dtype=tf.float32)

div = tf.divide(x, y)
print(div.numpy())

# 6. Tensor Reshaping
reshape_tensor = tf.constant([1, 2, 3, 4])

reshaped = tf.reshape(reshape_tensor, (2, 2))
print(reshaped.numpy())

# 7. Tensor Square
square_tensor = tf.constant([-2, -3, 4])

square = tf.square(square_tensor)
print(square.numpy())

# 8. Broadcasting Operation
broadcast_tensor = tf.constant([[1, 2], [3, 4]])

broadcast = broadcast_tensor + 5
print(broadcast.numpy())

# 9. Combining Tensors
t1 = tf.constant([[1, 2], [3, 4]])
t2 = tf.constant([[5, 6], [7, 8]])

combined = tf.concat([t1, t2], axis=0)
print(combined.numpy())

# 10. Advanced Element-wise Operations
p = tf.constant([[1, 2], [3, 4]])
q = tf.constant([[4, 3], [2, 1]])

maximum = tf.maximum(p, q)
print(maximum.numpy())

minimum = tf.minimum(p, q)
print(minimum.numpy())

absolute = tf.abs(tf.constant([[-1, -2], [3, -4]]))
print(absolute.numpy())

logarithm = tf.math.log(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
print(logarithm.numpy())

exponential = tf.exp(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
print(exponential.numpy())