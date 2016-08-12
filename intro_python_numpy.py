
# don't need to care about the first 3 lines (just some configuration for ipython notebook)
%matplotlib inline
import matplotlib
matplotlib.use('Agg')

from __future__ import print_function, division, absolute_import

import os
help(os)

import numpy as np
import matplotlib.pyplot as plt

## List
X = [1, 2, 3, 4]
print('Access list element:', X[0], X[1])
print('Slicing:', X[:2], X[1:3])
X[2:4] = [8, 9] # Assign a new sublist to a slice

## Dictionary
X = {'a': 1, 'b': 2}
print('Access dictionary element:', X['a'], X['b'])
print('Keys:', X.keys())
print('Values:', X.values())

## String
X = 'abcdxyz'
print('Access character:', X[0], X[1])
print(X.capitalize())  # Capitalize a string
print(X.upper())       # Convert a string to uppercase
print(X.rjust(7))      # Right-justify a string, padding with spaces
print(X.center(7))     # Center a string, padding with spaces
print(X.replace('abc', 'haha'))  # R

### Boolean
X = True
if X and False:
    print('You wont see anything here.')
if X or False:
    print('Now you see it')


if True:
    print("Do Something")
elif False:
    print("Nothing")
else:
    print("Anything")

animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)

# adding enumerate for indexing
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

# Full version
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)
# Short version
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)
# Short and fun version
nums = [0, 1, 2, 3, 4]
squares = [x**2 if x % 2 == 0
           else x**3
           for x in nums
           if x != 3]
print(squares)

d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, numb in d.iteritems(): # always use iteritems faster than items()
    print('A %s has %d legs' % (animal, numb))
for k in d.iterkeys():
    print(k)
for v in d.itervalues():
    print(v)

# Dictionary comprehension
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"

def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))

def multi_purpose(x, y=1):
    return x * y, x / y, x + y, x - y

a, b, c, d = multi_purpose(8)
print(a, b, c, d)

x = multi_purpose(8, y=12)
print(x)

def nestle():
    def abc():
        print('abc')
    return abc
nestle()()

f = lambda x: [x**2 for i in range(x)]
print(f(10))

class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method
g.greet(loud=True)   # Call an instance method

X = np.array(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9],
     [10,11,12]
    ], dtype='float32')
print('Basic info:', X.shape, X.dtype)

# Array indexing
print(X[0, :]) # first row, all columns
print(X[:, :2]) # all row, first 2 columns
print(X[:, -2:]) # all row, last 2 columns

# Transpose the matrix
print(X.T)

# Fastest way to shuffle or randomly sample from array
# (do not use np.random.shuffle very slow for big data)
idx = np.random.permutation(X.shape[0])
X = X[idx][:3] # sample first 3 samples

a = np.zeros((2,2))  # Create an array of all zeros
print(a)

b = np.ones((1,2))   # Create an array of all ones
print(b)

c = np.full((2,2), 7) # Create a constant array
print(c)

d = np.eye(2)        # Create a 2x2 identity matrix
print(d)

e = np.random.random((2,2)) # Create an array filled with random values
print(e)

x = np.array([[1,2],
              [3,4]], dtype=np.float64)
y = np.array([[5,6],
              [7,8]], dtype=np.float64)
z = np.array([[1,2,3],
              [4,5,6]], dtype=np.float64)

# Dot product
print(np.dot(x, y))

# Elementwise sum; both produce the array
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
print(np.sqrt(x))

print(np.sum(x))  # Compute sum of all elements;
print(np.sum(x, axis=0))  # Compute sum of each column;
print(np.sum(x, axis=1))  # Compute sum of each row

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
print(y)

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.figure()
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.

import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
