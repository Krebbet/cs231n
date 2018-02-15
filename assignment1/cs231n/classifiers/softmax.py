import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  N = y.shape[0] # number of samples
  D = X.shape[1] # input space dimension
  C = W.shape[1]  # number of classes
  for i in range(N):

    # First grab the scores
    scores = X[i,:].dot(W)
    # Correct for numerical stability
    scores -= np.max(scores)
    
    # get the denominator of our prob func.
    exp_sum = 0 
    for s in scores:
      exp_sum += np.exp(s)

    # now we can compute the loss function 
    # Li = -s_yi + log (sum ( exp(sj) ))
    loss += -scores[y[i]] + np.log(exp_sum)
    
    for j in range(C):
      # get gradient
      if (j == y[i]):
        dW[:,j] += X[i,:] * (np.exp(scores[j])/exp_sum - 1.0)
      else :
        dW[:,j] += X[i,:] * np.exp(scores[j]) / exp_sum
  
  # We want the mini-batch loss to be scaled to a single itt.    
  loss /= N
  dW /= N
  
  #regularization
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  
  N = y.shape[0] # number of samples
  D = X.shape[1] # input space dimension
  C = W.shape[1]  # number of classes

  

  # First we calculate scores S (N,C) 
  # W is broadcast to each set of inputs Xi
  # sij = Wj * xi -> Wj is the vector of lin predictors to class j
  scores = X.dot(W)
  exp_scores = np.exp(scores) 
  # should be N x C
  
  # We now grab a summed exponential likelihood vector
  sum_exp = np.sum(exp_scores,axis =1)
  # The shape should be (N,)
  
  # Calc the loss for each Li and sum them together
  # Li = -s_yi + ln( sum_j( exp(sj) ) )
  loss =  np.sum(-scores[np.arange(N),y]  +np.log(sum_exp),axis = 0)/N
  
  # Create a mask to add in the yi terms
  mask = np.zeros((N,C))
  mask[np.arange(N),y] = 1.0
  
  # Now we calculate the partial derivatives with respect to W
  # dW (D X C)
  # X (N x D)
  dW += X.T.dot(exp_scores/sum_exp.reshape(-1,1) - mask)/N
  # DxNxC
  
  

  
  # Add in regularization!
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
  
  
  
  return loss, dW

