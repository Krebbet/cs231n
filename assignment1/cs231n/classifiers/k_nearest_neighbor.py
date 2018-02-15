import numpy as np


class KNearestNeighbor:
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Input:
    X - A num_test x dimension array where each row is a test point.
    k - The number of nearest neighbors that vote for predicted label
    num_loops - Determines which method to use to compute distances
                between training points and test points.

    Output:
    y - A vector of length num_test, where y[i] is the predicted label for the
        test point X[i, :].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Input:
    X - An num_test x dimension array where each row is a test point.

    Output:
    dists - A num_test x num_train array where dists[i, j] is the distance
            between the ith test point and the jth training point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #dists2 = np.zeros((num_test, num_train))
    print('x',X.shape, self.X_train.shape)
    for i in range(num_test):
      for j in range(num_train):

        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]               #
        #####################################################################
        dists[i,j] = np.linalg.norm(X[i,:]-self.X_train[j,:])
        #dists2[i, j] = np.sqrt(np.sum((X[i, :] - self.X_train[j, :]) ** 2))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      dists[i,:] = np.linalg.norm(X[i,:]- self.X_train,axis = 1)
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      # X_train - X[i,:] -> this is the broadcasting partition
      # X[] is shape(length,) which is 1 row of train so it broad casts itself along
      # the train rows the result is a matrix of size X_train each row 
      # element subtracted by the vector Xi
      # dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=1)) # broadcasting
      #####################################################################
      #                       END OF YOUR CODE                            #
      #####################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 

    test_sum = np.sum(np.square(X),axis =1)
    train_sum = np.sum(np.square(self.X_train),axis =1)
    dists = np.sqrt(-2*X.dot(self.X_train.T) + test_sum.reshape(-1,1) + train_sum)
    
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # Output: sqrt((x-y)^2)
    # (x-y)^2 = x^2 + y^2 - 2xy -> get from inner products
    #test_sum = np.sum(np.square(X), axis=1) # num_test x 1
    #train_sum = np.sum(np.square(self.X_train), axis=1) # num_train x 1
    #inner_product = np.dot(X, self.X_train.T) # num_test x num_train
    #print(test_sum.shape)
    #dists = np.sqrt(-2 * inner_product + test_sum.reshape(-1, 1) + train_sum) # broadcast
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Input:
    dists - A num_test x num_train array where dists[i, j] gives the distance
            between the ith test point and the jth training point.

    Output:
    y - A vector of length num_test where y[i] is the predicted label for the
        ith test point.
    """
    # print('Start prediction!')
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      
      
      # First thing lets find the k closest points
      # We sort the distances into lowest to highest distance
      y_index = np.argsort(dists[i,:])
      #print('di shape:',dists[i,:].shape)
      # y_index holds an array ranking the enteries by dist low to highest
      # now we subset result the k lowest distances of known cases
      closest_y = self.y_train[y_index[:k]]
      #print('closest y shape:',closest_y.shape)
      # the training results are indexed by the first k entries in y_index (the k 
      # smallest distances
      # now we predict the result by choosing the category that came up most in 
      # the k closest predictions
      y_pred[i] = np.argmax(np.bincount(closest_y))
      
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # training point, and use self.y_train to find the labels of these      #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      # I dont get this axis = 0 declare....
      #y_indicies = np.argsort(dists[i, :], axis = 0)
      #print(y_indices)
      #closest_y = self.y_train[y_indicies[:k]]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #y_pred[i] = np.argmax(np.bincount(closest_y))
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred
'''
print("HELLO WORLD!")
x = KNearestNeighbor()

print("HELLO WORLD!")
x.train(5,5)
print("HELLO WORLD!")

'''