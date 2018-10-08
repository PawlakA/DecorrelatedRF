from collections import deque
import numpy as np
import imgutils

MIN_POINTS_NODE = 10
PURITY_THRESHOLD = 0.98
SIZE_REG = 5
NUM_CLASSES = 10
NUM_SPLITS = 5
TOP_SPLITS = 3
NUM_THRESHOLDS = 5
DECORRELATE = True
HEURISTICS_SPLIT_POINT = True
NAIVE_SPLIT_POINT = False

def splitPointSelection(features, dataset, dataLabels):
    """ Determine the split point theta_2 for a given feature.
    (The function performs different split point selections based on a defined
    global variable. This can be used for comparison.)
    Return: 3-tuple     (indices of datapoints forwarded to the left,
                        indices of datapoints forwarded to the right,
                        selected threshold theta_2)
    """
    d = (features[0] - features[1]) - (features[2] - features[3])
    # for naive split point threshold = 0
    if NAIVE_SPLIT_POINT:
        threshold = 0
        left_idx = d < threshold
        right_idx = ~left_idx
        return left_idx, right_idx, threshold
    # procedure described in the paper (Algorithm 2)
    if HEURISTICS_SPLIT_POINT:
        centroids = []
        for c in range(NUM_CLASSES):
            cls_idx = dataLabels == c
            if cls_idx.any():
                centroids.append(d[cls_idx].mean())
        centroids.sort() 
        largest_gap = np.diff(centroids).argmax()
        threshold = sum(centroids[largest_gap:largest_gap+2]) / 2
        left_idx = d < threshold
        right_idx = ~left_idx
        return left_idx, right_idx, threshold
    # sample NUM_THRESHOLDS threshold values between minimal and maximal value of d and pick one resulting in the biggest information gain
    else:
        thresholdRange = np.linspace(d.min(), d.max(), NUM_THRESHOLDS + 2)[1:NUM_THRESHOLDS + 1]
        threshold = thresholdRange[0]
        left_idx = d < threshold
        right_idx = ~left_idx
        infGain = infoGain(dataLabels, [(left_idx, right_idx)])
        for i in range(1,thresholdRange.shape[0]):
            left_idx_t = d < thresholdRange[i]
            right_idx_t = ~left_idx_t
            infoGain_t = infoGain(dataLabels, [(left_idx_t, right_idx_t)])
            if (infoGain_t > infGain):
                left_idx = left_idx_t
                right_idx = right_idx_t
                infGain = infoGain_t
                threshold = thresholdRange[i]
        return left_idx, right_idx, threshold

def testSplit(rows, cols, dataset, dataLabels, numOfRegions):
    '''
    Calculate feature value and decides whether data should be forwarded to the left or right child node.
    Return: the index for the left partition of the dataset and for the right partition, threshold and operator
    '''
    # choose random operator
    operators = np.array([0,1,2,3], dtype=np.uint8)
    if NAIVE_SPLIT_POINT:
        operator = 0
    else:
        operator = np.random.randint(len(operators))
    # calculate value of the feature
    features = np.zeros([4, dataset.shape[0]])
    for i in range(numOfRegions):
        features[i] = imgutils.calculateFeatureValue_new(dataset, [rows[i],cols[i]], SIZE_REG, func_id=operator)
    left_idx, right_idx, threshold = splitPointSelection(features, dataset, dataLabels)
    return left_idx, right_idx, threshold, operator

def entropy(data):
    '''
    Calculate entropy of data set based on the equation: sum over all classes(p(class)*log2(p(class)))
    Return: entropy of the dataset
    '''
    p_class = np.bincount(data) / data.size
    log_p_cl = np.log2(p_class, where=(p_class!=0), out=np.zeros_like(p_class))
    return -(p_class * log_p_cl).sum()

def infoGain(dataLabels, split_indices):
    '''
    Calculate information gain of the split based on the the equation: entropy before split - entropy of left
    labels * probability of goint to the left - entropy of right * probability of going to the right
    Return: information gain of the dataset
    '''
    infogain_diff = np.empty(len(split_indices))
    for i,(left_idx,right_idx) in enumerate(split_indices):
        left,right = dataLabels[left_idx],dataLabels[right_idx]
        infogain_diff[i] = left.size*entropy(left) + right.size*entropy(right)
    return entropy(dataLabels) - infogain_diff / dataLabels.size

def bestSplits(n, m, dataset, dataLabels):
    '''
    Sample n random pairs of regions and find m best pairs (in the sense of the biggest value of information gain).
    Return: 8-tuple   ( row-coordinates of m best splits,
                        column-coordinates of m best splits,
                        labels of data in child nodes after split for m best splits
                        thresholds of m best splits,
                        number of regions (1,2,4) of m best splits,
                        operators of m best splits,
                        row-coordinate of the best split,
                        column-coordinate of the best split)
    '''
    rows, cols, numOfRegions = imgutils.sampleRandomRegion(dataset.shape[1:], SIZE_REG, (n, 4))
    # Evaluate each one of them by splitting the data and computing info gain
    split_indices = np.empty((n, 2, dataset.shape[0]), dtype=np.bool)
    thresholds = np.empty(n)
    operators = np.empty(n)
    for i in range(n):
        left_idx, right_idx, thres, operator = testSplit(rows[i], cols[i], dataset, dataLabels, numOfRegions[i])
        split_indices[i, 0, :] = left_idx
        split_indices[i, 1, :] = right_idx
        thresholds[i] = thres
        operators[i] = operator
    info = infoGain(dataLabels, split_indices)
    # Return patches with higher information gain
    MmaxInfoIdx = np.argpartition(-info, m)[:m]
    OnemaxInfoIdx = np.argpartition(-info, m)[:1]
    return rows[MmaxInfoIdx], cols[MmaxInfoIdx], split_indices[MmaxInfoIdx], thresholds[MmaxInfoIdx], numOfRegions[MmaxInfoIdx], operators[MmaxInfoIdx], rows[OnemaxInfoIdx], cols[OnemaxInfoIdx]

def setSplit(tree, node_id, rows, cols, split, Y, threshold, numOfRegions, operator):
    '''
    Set the fields of the split node indicated by node_id and add the respective
    child nodes.
    '''
    lchild, rchild = node_id*2 + 1, node_id*2 + 2
    tree[:4, node_id] = rows
    tree[4:8, node_id] = cols
    tree[8, node_id] = numOfRegions
    tree[9, node_id] = threshold
    tree[10, node_id] = operator
    tree[11:13, node_id] = (lchild,rchild)
    # Left node
    ChildNodePlaceholder(tree, lchild, None, Y[split[0]])
    # Right node
    ChildNodePlaceholder(tree, rchild, None, Y[split[1]])

def splitNode(tree, node_id, trainingSet, trainingLabels, obj_function):
    '''
    Generate many splits and test each one with the objective function to be
    minimized.
    Return: 2-tuple (the left and right indices of the best split, statistics about decorrelation)
    '''
    # Generate several splits
    rows,cols,splits,thresholds,numOfRegions,operators,maxrows, maxcols = bestSplits(NUM_SPLITS, TOP_SPLITS, trainingSet, trainingLabels)
    # Determine the optimal one
    minimizer,minz = 0,np.inf
    if DECORRELATE and obj_function is not None:
        for i in range(rows.shape[0]):
            # Set this split, and estimate strength and correlation of the forest
            setSplit(tree, node_id, rows[i], cols[i], splits[i], trainingLabels, thresholds[i], numOfRegions[i], operators[i])
            # Calculate the measure we want to minimize
            z = obj_function(trainingSet, trainingLabels)
            if z < minz:
                minimizer,minz = i,z
    opt_split = splits[minimizer]
    # Set the optimal split into the tree
    setSplit(tree, node_id, rows[minimizer], cols[minimizer], opt_split,
             trainingLabels, thresholds[minimizer], numOfRegions[minimizer], operators[minimizer])
    # Check whether split point chosen using decorrelation procedure and split point selected based on information gain calculation differ
    decorrStats = 2
    if np.array_equal(rows[minimizer],maxrows[0]) and np.array_equal(cols[minimizer],maxcols[0]):
        decorrStats = 1
    return opt_split, decorrStats

def stopCriterion(labels, probabilities):
    '''
    Return: a bool to decide whether to stop growing the tree.
    '''
    return labels.size < MIN_POINTS_NODE or probabilities.max() > PURITY_THRESHOLD

def statsStopCriterion(leaf, labels):
    """ Collect statistics about the stop criterion.
    Return: -1 if the maximum heigth was reached,
            -2 if the minimum number of data points was achieved,
            -3 otherwise (i.e. purity criterion was met).
    """
    return -1 if leaf \
        else (-2 if labels.size < MIN_POINTS_NODE
              else -3)

def ChildNodePlaceholder(tree, node_id, data, labels):

    weights = 1/labels.size if labels.size else 0.
    probabilities = np.bincount(labels, minlength=NUM_CLASSES) * weights
    tree[:13, node_id] = 0
    tree[13:, node_id] = probabilities
    split = None,None
    return split

def addNode(tree, node_id, data, labels, leaf=False, obj_function=None):
    '''
    Add child node to the tree.
    Return: 2-tuple (split point, decorrelation statistisc)
    '''
    weights = 1/labels.size if labels.size else 0.
    probabilities = np.bincount(labels, minlength=NUM_CLASSES) * weights
    if leaf or stopCriterion(labels, probabilities):
        tree[:13, node_id] = 0
        tree[13:, node_id] = probabilities
        split = None,None
        decorrStats = statsStopCriterion(leaf, labels)
    else:
        split, decorrStats = splitNode(tree, node_id, data, labels, obj_function)
    return split, decorrStats


def classifier_margin(trees, X, Y):
    """ Return the predictions for each class given by each decision tree in the
    RF along with the margin and other measurements needed for the calculation
    of strength and correlation, given a trained RF and a dataset (X,Y).
    Return: 4-tuple    (predictions, Q(x,y), Q(x, \hat{j}), (p1,p2))
        (Consult the paper for the meaning of Q, \hat{j}, p1, p2)
    """
    preds, votes = predictFromForest(X, trees)
    # Calculate Q(x,y) and Q(x,j_hat)
    Q = votes.mean(axis=0)
    Qxy = Q[range(Y.shape[0]), Y]
    # Trick: set the proportion of votes to the true label y to -1
    # in order to compute max_{j != y} P(h(X, Omega) = j)
    Q[range(Y.shape[0]), Y] = -1
    j_hat = Q.argmax(axis=1)
    Qxj_hat = Q[range(j_hat.shape[0]), j_hat]
    # Calculate the mean of votes over the dataset for every tree
    p = np.empty((2, votes.shape[0]))
    p[0] = votes[:, range(Y.shape[0]), Y].mean(axis=1)
    p[1] = votes[:, range(j_hat.shape[0]), j_hat].mean(axis=1)
    return preds, Qxy, Qxj_hat, p

def out_of_bag_prediction(trees, subsets, X_train, Y_train):
    """ Given a trained RF, the bootstrap datasets of each tree, and the
    training dataset (X_Train, Y_Train), compute the out-of-bag predictions of
    each decision tree. (Used for evaluation and plot generation.)
    Return: 5-tuple (indices of OOB data points for each tree,
                     OOB predictions for each tree,
                     Q(x,y) of each tree,
                     Q(x, \hat{j}) of each tree,
                     p1,p2 of each tree).
        (Consult the paper for the meaning of Q, \hat{j}, p1, p2)
    """
    oob_predictions = np.empty((X_train.shape[0], NUM_CLASSES))
    oob_idx = np.zeros(X_train.shape[0], dtype=np.bool)
    Qxy = np.empty(X_train.shape[0])
    Qxj_hat = np.empty(X_train.shape[0])
    p = np.zeros((3, trees.shape[0]))
    for i,x in enumerate(X_train):
        # Which classifiers don't have x in its training set
        tree_idx = ~((subsets == x).all(axis=(2,3)).any(axis=1))
        oob_trees = trees[tree_idx]
        if oob_trees.shape[0]:
            oob_idx[i] = True
            preds_x,Qxy_x,Qxj_x,p_x = classifier_margin(oob_trees, x[np.newaxis],
                                                        Y_train[i,np.newaxis])
            oob_predictions[i] = preds_x[0]
            Qxy[i] = Qxy_x[0]
            Qxj_hat[i] = Qxj_x[0]
            # Accumulate the votes of each out-of-bag tree of this data point
            p[:2,tree_idx] += p_x
            p[2,tree_idx] += 1
    # To make the average of votes of each tree, divide by the number of
    # datapoints on which each the tree was evaluated
    idx_evaluated_trees = p[2] != 0
    np.divide(p[0], p[2], out=p[0], where=idx_evaluated_trees)
    np.divide(p[1], p[2], out=p[1], where=idx_evaluated_trees)
    return oob_idx, oob_predictions, Qxy, Qxj_hat, p

def strength_correlation(Qxy, Qxj_hat, p, Y):
    """ Calculate strength and correlation of e RF based on Breiman's formulas.
    Return: 2-tuple     (strength, correlation)
    """
    # mr = P(h(X, Omega) = Y) - max_{j != y} P(h(X, Omega) = j)
    mr = Qxy - Qxj_hat
    # Calculate the strength as the average of the margin over the dataset
    s = mr.mean()
    # Variance of the margin
    var = (mr**2).mean() - s**2
    # Standard deviation
    E_sd = np.sqrt(p[0] + p[1] + (p[0] - p[1])**2).mean()
    # Correlation
    corr = var / E_sd**2
    return s, corr

def c_s2_ratio(trees, X_train, Y_train, upperbound):
    """ Calculate the c/s2 ratio of a RF given a dataset (X_train, Y_train) and
    the upperbound for the case of nonpositive strength.
    Return: float   c/s2 ratio
    """
    _,Qxy,Qxj_hat,p = classifier_margin(trees, X_train, Y_train)
    s,ro = strength_correlation(Qxy, Qxj_hat, p, Y_train)
    if s > 0:
        return ro / s**2
    else:
        return upperbound - s

def fit(X_train, Y_train, num_trees, height, sample_size):
    """ The training procedure. Given the training dataset (X_Train, Y_Train),
    number of trees in the RF, maximum allowed height, and bootstrap dataset
    size, execute the training method and return the trained trees.
    Return: 3-tuple     (trees, boostrap datasets, RF statistics)
    """
    # Set up the memory and data structure
    trees = np.zeros((num_trees, NUM_CLASSES + 13, 2**height - 1))
    decorrStatistics = np.zeros((num_trees, 2**height - 1))
    subsets = np.empty(
        (num_trees, sample_size, X_train.shape[1], X_train.shape[2]),
        dtype=X_train.dtype)
    labs = np.empty((num_trees, sample_size), dtype=Y_train.dtype)
    queue = deque()
    # Set up the objective function being optimized at the decorrelation
    def obj_f(X, Y):
        upperbound = num_trees**2 * (NUM_CLASSES-1) * (num_trees*X.shape[0])**2
        return c_s2_ratio(trees, X, Y, upperbound)
    # Create the root of every tree
    for i in range(num_trees):
        idx = np.random.choice(X_train.shape[0], sample_size)
        subsets[i] = X_train[idx]
        labs[i] = Y_train[idx]
        splitidx, decorrStats = addNode(trees[i], 0, subsets[i], labs[i])
        decorrStatistics[i,0] = decorrStats
        lft_idx = splitidx[0,:]
        rght_idx = splitidx[1,:]
        queue.extend([lft_idx, rght_idx])
    # Grow them breadth-first
    for h in range(1, height):
        breadth = 2**h
        last_id = breadth - 1
        leaf = h == height-1
        for i in range(num_trees):
            for j in range(breadth):
                node_data_idx = queue.popleft()
                if node_data_idx is None:
                    subset_left,subset_right = None,None
                else:
                    d = subsets[i][node_data_idx]
                    l = labs[i][node_data_idx]
                    splitidx, decorrStats = addNode(trees[i], last_id+j, d, l,
                                                leaf=leaf, obj_function=obj_f)
                    decorrStatistics[i, last_id+j] = decorrStats
                    if splitidx[0] is None and splitidx[1] is None :
                        lft_idx, rght_idx = None, None
                    else:
                        lft_idx = splitidx[0, :]
                        rght_idx = splitidx[1,:]
                    subset_left = node_data_idx.copy()
                    subset_left[node_data_idx] = lft_idx
                    subset_right = node_data_idx.copy()
                    subset_right[node_data_idx] = rght_idx
                queue.extend([subset_left, subset_right])
    return trees,subsets, decorrStatistics

def forwardData(rows, cols, dataset, numOfRegions, threshold, operator):
    '''
    Decide whether data should be forwarded to the right of left child node.
    Return: labels of data in the left and right child node
    '''

    features = np.zeros([dataset.shape[0], 4])
    for i in range(4):
        if (i<numOfRegions):
            features[:, i] = imgutils.calculateFeatureValue_new(dataset, [rows[i],cols[i]], SIZE_REG, func_id=operator)
    d = (features[:, 0] - features[:, 1]) - (features[:, 2] - features[:, 3])
    left_idx = d < threshold
    right_idx = ~left_idx
    return left_idx, right_idx

def predictFromTree(images, tree, node_idx=0):
    '''
    Predict the label using one tree
    Return:     ndarray with shape (N,C) with probabilities for each data point,
                where N is the number of data points and C the number of classes.
                Or None in case the dataset is empty
    '''
    # If there is no image to test
    if images.shape[0] == 0:
        return None
    # If it is leaf node
    node = tree[:,node_idx]
    if (node[:13] == 0).all():
        return node[13:]
    # Split the images set
    rows = node[:4].astype(np.int)
    cols = node[4:8].astype(np.int)
    numOfRegions = node[8]
    thresh = node[9]
    operator = node[10]
    lft_idx,rght_idx = forwardData(rows, cols, images, numOfRegions, thresh, operator)
    # Recursively predict at the children nodes
    result = np.empty((images.shape[0],NUM_CLASSES))
    result[lft_idx] = predictFromTree(images[lft_idx], tree, int(node[11]))
    result[rght_idx] = predictFromTree(images[rght_idx], tree, int(node[12]))
    return result

def predictFromForest(images, trees):
    '''
    Predict the label using the whole forest
    Return:     ndarray with shape (N,C) with probabilities for each data point,
                where N is the number of data points and C the number of classes.
                Or None in case the dataset is empty
    '''
    numberOfTrees = len(trees)
    N = images.shape[0]
    predictions = np.zeros((N, NUM_CLASSES))
    votes = np.zeros((numberOfTrees, N, NUM_CLASSES), np.bool)
    for i,tree in enumerate(trees):
        # Accumulate probabilities
        pred = predictFromTree(images, tree)
        predictions += pred
        # Record the vote of this tree for every data point
        pclass = pred.argmax(axis=1)
        votes[i][range(N), pclass] = True
    return predictions / numberOfTrees, votes

