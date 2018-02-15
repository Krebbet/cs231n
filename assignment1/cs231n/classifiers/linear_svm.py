import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_var = X.shape[1]
  #print('classes =',num_classes,'training ex. = ',num_train,'input size =',num_var)
  loss = 0.0
  for i in range(num_train):
    # Forward Prop
    scores = X[i].dot(W) # Find the scores generated for each class...
    correct_class_score = scores[y[i]] #Records the score of the correct class only.
    for j in range(num_classes):
      if j == y[i]:
        continue
      # If the score of class j is close enough to the correct score (within 1) record 
      # it as a lose.
      
      # Calculate Lose
      # If score i is within 1 of the correct class then penalize the network...
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # Calculate the derivative of Wj
        dW[:,j] += X[i,:].T
        dW[:,y[i]] -= X[i,:].T        
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss. ie. penalize sum of weights....
  loss += 0.5 * reg * np.sum(W * W)

  # Add regularization to grad.
  dW += reg*W
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  # Write out the simplified back prop equation and implement!!!
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train =  X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  # First we compute the scores of each input xi
  # NxC
  scores = np.dot(X,W)
  #print('scores shape',scores.shape)
  #print(scores)
  # The result should be a NxC array storing the scores of the
  # nth input with scores k exist in C
  
  # next we compute a margin matrix,
  
  # first we get the corect scores vectorized
  # N
  cor_scores = scores[np.arange(num_train),y]
  #print('cor scores shape',cor_scores.shape)
  #print('transpose cor scores shape',cor_scores.reshape(-1,1).shape)

  
  # then we get the margins
  # cor_scores needs to be reshaped into a row vector...
  # NxC
  margin = np.maximum(0,scores - cor_scores.reshape(-1,1) + 1.0)
  #print('margin shape',margin.shape)
  
  # Set the correct classes to zero 
  margin[np.arange(num_train),y] = 0
  #print(margin)
  ones = np.zeros(margin.shape)
  ones[margin > 0 ] = 1
  incorrect_count= np.sum(ones, axis=1)
  #print(incorrect_count)
  ones[np.arange(num_train),y] = -incorrect_count

  
  dW = X.T.dot(ones)
  #print(dW)
  
  #print(X.T.shape)
  #print(np.sum(ones,axis=1).shape)
  #print(np.sum(ones,axis=1).reshape(-1,1).shape)
  # now I need to add in the negative terms
  #dW[np.arange(num_train),y] -= X.T.dot(np.sum(ones,axis=1))
  #print(X.shape,np.sum(-ones,axis=1).shape)
  # This is a vector which gives the dw/dyi for each data entry. (length data.)
  #(X * (np.sum(-ones,axis=1).reshape(-1,1) ) ).T
  
  # Sum everything up for the loss
  loss = np.sum(margin) / num_train
  dW /= num_train
  
  # Add in the regularization error
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W

  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
