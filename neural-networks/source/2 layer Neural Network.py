#2 layer Neural Network: Input layer=1; Output layer:1 Hidden layers=0 
import numpy as np

# sigmoid function: Used to compute probabilities coz gives result in between 0 and 1.
def nonlin(x,deriv=False):#x is a 4X1 matrix
    if(deriv==True):#Generating derivative when deriv=True
        return x*(1-x)
    return 1/(1+np.exp(-x))#Computes sigmoid for each element in X

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])#Represents 3 input nodes and 1 output node.

# output dataset
y = np.array([[0,0,1,1]]).T#Thats the transpose matrix of y, it converts y into a 4X1 matrix, as desired.

# seed random numbers to make calculation deterministic (just a good practice)-Randomly distributes the number but in the same way, each and every time.
np.random.seed(1)
# Weights Matrix: Syn0 or Synapse0:This matrix is the neural network, all learning stored here.
# initialize weights randomly with mean 0
# Dimensions of this matrix are: 3X1 coz layer0 has 3 nodes and layer1 has 1 node.
syn0 = 2*np.random.random((3,1))- 1
# Actual Network training code.
for iter in range(10000):
    # forward propagation
    # Full batch training, processing all training examples at the same time.
    l0 = X
    # Input to layer1=Sigmoid of output from layer0=Sigmoid of(Input to layer0 * Synapse matrix b/w layer 0 and layer 1)
    # np.dot(l0, syn0): Forms the matrix product.And nonlin() computes the Sigmoid.
    # Notice the dimensions of the matrix: 4X3 and 3X1, thus a 4X1 matrix.
    l1 = nonlin(np.dot(l0,syn0))
    # how much did we miss?
    l1_error = y - l1
    # multiply how much we missed by the slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)
    # update weights of the cost function syn0; syn0=syn0+Transpose(l0)*(l1_delta)
    syn0 += np.dot(l0.T,l1_delta)
print("Output After Training:")
print(l1)
