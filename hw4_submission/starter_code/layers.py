import numpy as np

def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    
    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).
    
    Inputs:
    - x: A numpy array containing input data, of shape (N, Din)
    - w: A numpy array of weights, of shape (Din, Dout)
    - b: A numpy array of biases, of shape (Dout,)
    
    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = np.matmul(x,w) + b
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    
    Inputs:
    - dout: Upstream derivative, of shape (N, Dout)
    - cache: returned by your forward function. Tuple of:
      - x: Input data, of shape (N, Din)
      - w: Weights, of shape (Din, Dout)
      - b: Biases, of shape (Dout,)
      
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, Din)
    - dw: Gradient with respect to w, of shape (Din, Dout)
    - db: Gradient with respect to b, of shape (Dout,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = np.matmul(dout,w.T) 
    dw = np.matmul(x.T,dout)
    N,N1 = np.shape(dout)
    db = np.matmul(dout.T,np.ones((N,)))
       
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # if all(x)>=0:
    #     out = x
    # else:
    #     out = np.zeros(np.shape(x))
    #     out =0

    # out = np.array([max(0,i) for i in x])
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: returned by your forward function. Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # dx = np.zeros(np.shape(x))
    # for i, j in enumerate(x):
    #     if j>=0:
    #         dx[i] = 1
    #     else:
    #         dx[i] = 0
    # dx = np.matmul(dout,dx)  
    x=np.maximum(0,x)
    x[x>0]= 1

    # dx = dout*dx      
    dx = dout*x    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def l2_loss(x, y):
    """
    Computes the loss and gradient of L2 loss.
    loss = 1/N * sum((x - y)**2)

    Inputs:
    - x: Input data, of shape (N, D)
    - y: Output data, of shape (N, D)

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement L2 loss                                                 #
    ###########################################################################
    N,D = np.shape(x)
    loss = 1/N * np.sum((x-y)**2)
    # dx = 1/N * np.sum(2*(x-y))
    dx = 1/N*2*(x-y)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

def softmax_loss(x, y):
   """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
   loss, dx = None, None
    ###########################################################################
    # TODO: Implement softmax loss                                            #
    ###########################################################################
   loss = 0 
   dx = 0

   N= np.shape(x)

   # if len(N)==1:
   #  N = N[0]
   #  max_arr = np.max(x)
   #  sumar =np.sum(np.exp(x-max_arr))
   #  yi = np.exp(x-max_arr)/sumar

   #  loss = - np.log(yi[y[0]])
   #  # loss = 1/N*loss
   #  # m = y.shape[0]
   #  yi[y] = yi[y]-1
   #  # yi = yi
   #  dx = yi
   # else:
   N = N[0]
   max_arr = np.max(x,axis=1)
   max_arr = max_arr[:,None]
   sumar =np.sum(np.exp(x-max_arr),axis=1)
   sumar = sumar[:,None]
   yi = np.exp(x-max_arr)/sumar
   for i in range(N):
    loss += - np.log(yi[i][y[i]])
   loss = 1/N*loss
   m = y.shape[0]
   yi[range(m),y] = yi[range(m),y]-1
   yi = yi/m
   dx = yi
   # sumar =np.sum(np.exp(x),axis=1)
   # sumar =np.sum(np.exp(x-max_arr),axis=1)

   # sumar = sumar[:,None]
   # yi = np.exp(x)/sumar
   # yi = np.exp(x-max_arr)/sumar

   


   # loss = 1/N*loss

   # m = y.shape[0]
   # yi[range(m),y] = yi[range(m),y]-1
   # yi = yi/m
   # dx = yi

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
   return loss, dx

