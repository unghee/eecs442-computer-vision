import pickle
import numpy as np
from layers import *

class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3072, hidden_dim=None, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer, None
          if there's no hidden layer.
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.weight_scale = weight_scale
        self.hidden_dim = hidden_dim
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights and biases using the keys        #
        # 'W' and 'b', i.e., W1, b1 for the weights and bias in the first linear   #
        # layer, W2, b2 for the weights and bias in the second linear layer.       #
        ############################################################################
        
        mu, sigma = 0, weight_scale


        if hidden_dim is None:
            W = np.random.normal(mu, sigma, (input_dim,num_classes))
            b = np.zeros((num_classes,))
            self.params = {'W': W, 'b' : b}

        else:   
            W1 = np.random.normal(mu, sigma, (input_dim,hidden_dim))
            b1 = np.zeros((hidden_dim,))

            W2 = np.random.normal(mu, sigma, (hidden_dim,num_classes))
            b2 = np.zeros((num_classes,))


            self.params = {'W2': W2, 'b2' : b2,'W1': W1, 'b1' : b1}




        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def forwards_backwards(self, X, y=None, return_dx = False):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, Din)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass. And
        if  return_dx if True, return the gradients of the loss with respect to 
        the input image, otherwise, return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################


        if self.hidden_dim is None:
            scores, cache = fc_forward(X,self.params['W'],self.params['b'])
        else:
            y_1, cache1 = fc_forward(X,self.params['W1'],self.params['b1'])
            y_2, cache2 = relu_forward(y_1)
            scores, cache3 = fc_forward(y_2,self.params['W2'],self.params['b2'])
 

        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   # 
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        if self.hidden_dim is None:

            loss_soft,dloss_soft =softmax_loss(scores,y) 
            loss = loss_soft + 0.5*self.reg*np.linalg.norm(self.params['W'])**2
            # loss = loss_soft + 0.5*self.reg*np.linalg.norm(W)**2
            dx, dW, db = fc_backward(dloss_soft,cache)
            dW = dW + self.params['W']*self.reg
            # dW = dW + W*self.reg
            grads= {'W': dW,'b': db}

        else:


            # calculate loss
            loss,dloss_soft =softmax_loss(scores,y)

            # backward fc1
            dx2, dW2, db2 = fc_backward(dloss_soft,cache3)

            # backward relu
            dx_relu = relu_backward(dx2,cache2)

            # backward fc2
            dx1, dW1, db1 = fc_backward(dx_relu ,cache1)


            grads= {'W1': dW1,'b1': db1,'W2': dW2,'b2': db2}









        

        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        if return_dx and self.hidden_dim is None:
            return dx
        elif return_dx and self.hidden_dim is not None:
            return dx1, scores
        else:
            return loss, grads

    def save(self, filepath):
        with open(filepath, "wb") as fp:   
            pickle.dump(self.params, fp, protocol = pickle.HIGHEST_PROTOCOL) 
            
    def load(self, filepath):
        with open(filepath, "rb") as fp:  
            self.params = pickle.load(fp)  
