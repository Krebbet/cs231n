import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  
  # 'flatten' the x data into N input vectors
  xp = np.reshape(x,(x.shape[0],-1))
  
  
  # calculate the forward pass.
  out = xp.dot(w) + b
  
  # cache the nec. values...
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None

  xp = np.reshape(x,(x.shape[0],-1)).T
  dw = np.dot(xp,dout)
  dx = np.dot(dout,w.T).reshape(x.shape)
  db = np.sum(dout,axis = 0)
  
  # reg????
  
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
  out = np.maximum(0,x)

  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache

  dx = np.zeros_like(x)
  dx[x > 0] = 1
  dx = dx*dout
  
  
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:
  
  note: momentum = decay....
  
  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    
    # calculate the mean and variance over the mini-batch for each feature
    #mean = np.sum(x,axis=0)/N
    mean = np.mean(x,axis = 0)
    r = x - mean # store the residuals for cache
    #var = (np.sum(r**2,axis = 0)/N)
    var = np.var(x,axis =0)
    #moment_2 = np.sum(x*x,axis=0)/N #<x^2>
    #var = np.sqrt(moment_2 - mean**2) #<x^2> -<x>^2

    
    # apply the normalization
    x_norm =  r/(np.sqrt(var + eps))
    
    
    # apply learned parameters
    out = gamma*x_norm + beta
    
    #update the running mean and var...
    running_mean = momentum*running_mean + (1.0 - momentum)*mean
    running_var = momentum*running_var + (1.0 - momentum)*var
    
    #print(bn_param['running_mean'],bn_param['running_var'])
 
    # what do I need in cache?
    # well figure out the back pass fuck face
    cache = {}
    cache['x_norm'] = x_norm
    cache['var'] = var
    cache['eps'] = eps
    cache['gamma'] = gamma
    cache['r'] = r
    cache['N'] = N
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    
    # apply the running mean/var for normalization
    # apply the normalization
    x_norm =  (x - running_mean)/(np.sqrt(running_var)+ eps)
    
    
    # apply learned parameters
    out = gamma*x_norm + beta    
    
    # nothing needed in cache...
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  # dyi/d?
  
  
  r = cache['r']
  var = cache['var']
  N = cache['N']
  denom = cache['var'] + cache['eps']
  dydbeta = 1.0
  
  dldy = dout
  dldgamma = np.sum(dldy*cache['x_norm'],axis = 0)
  dldbeta = np.sum(dldy,axis = 0)
  
  dldn = cache['gamma']*dldy # check
  
  
  dndv = -0.5*r*(denom**(-3/2)) # check
  dndu = -1.0/np.sqrt(denom) #check
  
  dldv = np.sum(dldn*dndv,axis = 0) #check
  dvdu = (-2.0/N)*np.sum(r,axis = 0) #check
  
  dndx = 1.0/np.sqrt(denom)
  dvdx = (2.0/N)*r
  
  dudx =1.0/N
  
  dldu = np.sum(dldn*dndu,axis = 0) + dldv*dvdu #check
  
  dldx = dldn*dndx + dldv*dvdx + dldu*dudx
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  dx = dldx
  dgamma = dldgamma
  dbeta = dldbeta
  #print(dbeta,dgamma)
  
  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < (1.0-dropout_param['p'])) / (1.0-dropout_param['p'])
    out = mask*x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    # Do nothing
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = mask*dout
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  
  # note: We input a 3d volume into the weights...
  # ie. the color space (or channels) or feaeture space is 
  # part of the input into the filter in this implementation.
  
  # copy the parameters over to easier read labels
  N, C, H, W = x.shape 
  F, _, HH, WW = w.shape
  stride = conv_param['stride']
  zero_pad = conv_param['pad']
  
  # Calculate output size and create the output array.
  H_out =int(1 + (H + 2 * zero_pad - HH) / stride)
  W_out = int(1 + (W + 2 * zero_pad - WW) / stride)
  #print(H_out,W_out)
  out = np.zeros((N,F,H_out,W_out))
  
  # pad our input
  H_pad = H + 2*zero_pad
  W_pad = W + 2*zero_pad
  x_pad = np.zeros((N,C,H_pad,W_pad))
  x_pad[:,:,zero_pad:(H_pad -zero_pad),zero_pad:(W_pad -zero_pad)] = x
  
  
  for n in range(N):
    for k in range(F):
      for i in range(0,H_pad - HH +1,stride):
        for j in range(0,W_pad - WW +1,stride):
          i_out = i//stride
          j_out = j//stride
          out[n,k,i_out,j_out] = np.sum(w[k,:,:,:]*x_pad[n,:,i:i+HH,j:j+WW]) + b[k]
          
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param, x_pad)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # grab local copies of all the nec. vars
  x, w, b, conv_param, x_pad= cache
  stride = conv_param['stride']
  zero_pad = conv_param['pad']
  # grab the shapes of the in middle and out
  N, C, H, W = x.shape 
  F, _, HH, WW = w.shape  
  
  # for simplicity project the zero padded input gradient and then grab what you want.
  dldx_pad = np.zeros_like(x_pad)
  
  
  #define vars for understanding
  dldy = dout
  # This will have shape (N, F, H', W')
  # where 
  #  H' = 1 + (H + 2 * pad - HH) / stride
  #  W' = 1 + (W + 2 * pad - WW) / stride  
  H_pad = H + 2*zero_pad
  W_pad = W + 2*zero_pad  

  
  # - x: Input data of shape (N, C, H, W)
  # - w: Filter weights of shape (F, C, HH, WW)
  dldw = np.zeros_like(w)

  
  #print('***************')
  #print(stride)
  #print(zero_pad)
  #Calculate the dldw gradients
  for k in range(F):
    for i in range(0,H_pad - HH + 1,stride):
      for j in range(0,W_pad - WW + 1,stride):
        x_contrib = x_pad[:,:,i:i+HH,j:j+WW]
        #print(x_contrib.T.shape)
        #print(k,i,j)
        #print(dldy[:,k,i,j].shape)
        #print(dout.shape)
        dldw[k,:,:,:] += np.dot(x_contrib.T,dldy[:,k,i//stride,j//stride]).T
    

    
    
  # Calculate the dldx grads... these I just reverse engineered.  
  for n in range(N):
    for k in range(F):
      for i in range(0,H_pad - HH +1,stride):
        for j in range(0,W_pad - WW +1,stride):
          i_out = i//stride
          j_out = j//stride
          dldx_pad[n,:,i:i+HH,j:j+WW] += dldy[n,k,i_out,j_out]*w[k,:,:,:]
          #out[n,k,i_out,j_out] = np.sum(w[k,:,:,:]*x_pad[n,:,i:i+HH,j:j+WW]) + b[k]    
  
  db = np.sum(dldy,axis=(0,2,3))
  dx = dldx_pad[:,:,zero_pad:(H_pad -zero_pad),zero_pad:(W_pad -zero_pad)]
  dw = dldw

  
  
  


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # Clearly this code assumes the architecture does not give a pooling size that
  # will index out of bounds of the feature layer from the pooling size
  
  # grab local copies of params
  
  N, C, H, W = x.shape
  h = pool_param['pool_height']
  w = pool_param['pool_width']
  s = pool_param['stride']
  
  Hpool = H//h
  Wpool = W//w
  out = np.zeros((N,C,Hpool,Wpool))
  for n in range(N):
    for i in range(0,H,h):
      for j in range(0,W,w):
        z = np.amax(x[n,:,i:i+h,j:j+w],axis =(1,2))
        out[n,:,i//h,j//w] = z
        
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  
  N, C, H, W = x.shape
  h = pool_param['pool_height']
  w = pool_param['pool_width']
  s = pool_param['stride']  


  Hpool = H//h
  Wpool = W//w
  dx = np.zeros_like(x)
  for n in range(N):
    for i in range(0,H,h):
      for j in range(0,W,w):
        z = x[n,:,i:i+h,j:j+w]
        # find the max val in the patch
        max_val = np.ndarray.max(z,axis = (1,2))

        # create a mask 
        mask = np.zeros_like(z)
        for s in range(C):
          mask[s] = (z[s,:,:] == max_val[s])*dout[n,s,i//h,j//w]
        
        # input the derivative into the proper hole, the rest are 0.
        dx[n,:,i:i+h,j:j+w] += mask
        
    
  
  
   
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = x.shape
  
  # we reshape the input to be (NxHxW,C)
  flat_x = np.transpose(x,(0,2,3,1))
  flat_x = flat_x.reshape((N*H*W,C))
  
  # push through the 'vanilla' batch norm
  flat_out, cache = batchnorm_forward(flat_x, gamma, beta, bn_param)
  
  # reshape the output back into the origional shape.
  out = flat_out.reshape((N,H,W,C))
  out = out.transpose((0,3,1,2))
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  
  N, C, H, W = dout.shape
  
  # we reshape the input to be (NxHxW,C)
  dout_flat = np.transpose(dout,(0,2,3,1))
  dout_flat = dout_flat.reshape((N*H*W,C))
  
  # push through the 'vanilla' batch norm
  dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
  
  # reshape the output back into the origional shape.
  dx = dx_flat.reshape((N,H,W,C))
  dx = dx.transpose((0,3,1,2))  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
