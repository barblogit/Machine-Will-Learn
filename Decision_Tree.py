"""
Part of the code is from MPCS 53111 by Professor Amitabh Chaudhary. Please do not Plagiarize.  
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mpl.rc('figure', figsize=[12,8])  # set the default figure size


"""
We use the arrythmia dataset: http://archive.ics.uci.edu/ml/datasets/Arrhythmia
"""
df = pd.read_csv('./arrhythmia.data', header=None, na_values="?")

"""
Class for decision tree nodes
"""
class Node(object):

	def __init__(self):
		self.name = None
		self.node_type = None # e.g. node_type = 'leaf'
		self.label = None # if node is a leaf, its label is the class label
		self.data = None
		self.split = None
		self.children = [] # each element in the list contains data for a child node


	def __repr__(self):
		# Return the representation of the current node as a string
		data = self.data
		if self.node_type != 'leaf':
			s = (f"{self.name} Internal node with {data[data.columns[0]].count()} rows; split"
                 f" {self.split.split_column} at {self.split.point:.2f} for children with"
                 f" {[p[p.columns[0]].count() for p in self.split.partitions()]} rows"
                 f" and infomation gain {self.split.info_gain:.5f}")
		else:
			s = "testing"
			s = (f"{self.name} Leaf with {data[data.columns[0]].count()} rows, and label"
                 f" {self.label}")
		return s


"""
Class for splits - stores information about a split (on a continuous or binary attribute).
"""
class Split(object):

    def __init__(self, data, class_column, split_column, point=None):
        self.data = data
        self.class_column = class_column
        self.split_column = split_column
        self.info_gain = None
        self.point = point         # spliting point
        self.partition_list = None # stores the data points on each side of the split
        self.compute_info_gain()


    def compute_info_gain(self):
        # sort data by split_column, in ascending order
        self.data = self.data.sort_values(by=self.split_column)

        # class array and split label array
        class_label_array = self.data[self.class_column]
        split_label_array = self.data[self.split_column]

        # count the number for first split-- same as the parent entropy
        C = class_label_array.value_counts().sort_index() # count for each unique label
        K = sorted(list(set(class_label_array))) # unique labels sorted in asc order
        prev_gt = dict(zip(K,C))                 # {split1: count1, split2: count2, ...}
        prev_le = dict(zip(K,[0]*len(K)))        # {split1: 0,      split2, 0,      ...}
        
        E_parent = self.entropy(prev_gt, prev_le) # parent entropy
        G_0   = 0   # for first split, info_gain is 0
        G_max = 0   # max gain
        
        # an np array of unique split values
        ori_val = split_label_array.unique() 

        # case 1: the attribute is binary, then split point is 0.5.
        # we can call self.entropy() directly to get info_gain
        if len(ori_val) == 2:
            # for binary attribute, we just need one step to update prev_gt and prev_le
            # alternatively, we can just do a count directly
            zero_dict = class_label_array[split_label_array == 0].value_counts().sort_index()
            one_dict  = class_label_array[split_label_array == 1].value_counts().sort_index()

            entropy = self.entropy(dict(zip(K,zero_dict)), dict(zip(K,one_dict)))
            gain = E_parent - entropy
            self.info_gain = gain
            self.point = 0.5

        # case 2: the attribute is continuous, then find split point of max info-gain
        # 	step 1: get the mid point list
        else:
	        tmp_val = np.insert(ori_val, 0, ori_val[0]-1) # helper array
	        mid_val = [(ori_val[i] + tmp_val[i])/2 for i in range(len(ori_val))]
	        mid_val = np.insert(mid_val, len(mid_val), mid_val[-1]+1)

	        # count for continuous info_gain
	        for i in range(len(mid_val)-1):
	            # traverse self.data, change corresponding dict by the class value
	            # notice ori_val[i] is the only unconsidered datapoint value at any step...
	            # that fall to the left of the NEXT midpoint
	            indices = class_label_array[split_label_array == ori_val[i]].values
	            # hence we increment le and decrement gt
	            for index in indices:
	                prev_gt[index] -= 1
	                prev_le[index] += 1
	            
	            G_cur = E_parent - self.entropy(prev_gt, prev_le)
	            if G_cur > G_max:
	                G_max = G_cur
	                self.point = mid_val[i+1] # split point is the NEXT midpoint
	        
	        self.info_gain = G_max

    
    # d1:dict({classLabel: count})
    def entropy(self, d1, d2):
        # entropy will be weighted avg of S1 and S2, 
        # weighted by proportion of examples in d1 and proportion of examples in d2
        sum_d1 = sum(d1.values()) #n
        sum_d2 = sum(d2.values()) #p
        sum_np = sum_d1 + sum_d2  #n+p
        S1 = S2 = 0

        if(sum_d1):
            for v in d1.values():
                if(v):
                    p = v/sum_d1
                    S1 += p * np.log2(1/p) # compute the entropy value
            S1 *= sum_d1/sum_np
        if(sum_d2):
            for v in d2.values():
                if(v):
                    q = v/sum_d2
                    S2 += q * np.log2(1/q) # compute the entropy value
            S2 *= sum_d2/sum_np
        return S1 + S2

                    
    """Get the two partitions (child nodes) for this split."""
    def partitions(self):
        if self.partition_list:
            # This check ensures that the list is computed at most once.
            # Once computed it is stored
            return self.partition_list
        
        data = self.data
        split_column = self.split_column
        partition_list = []
        partition_list.append(data[data[split_column] <= self.point])
        partition_list.append(data[data[split_column] >  self.point])

        return partition_list


"""
Class for decision tree
"""
class DecisionTree(object):

    def __init__(self, max_depth=None):
        if (max_depth and (max_depth != int(max_depth) or max_depth < 0)):
            raise Exception("Invalid max depth value.")
        self.max_depth = max_depth
    

    """Fit a tree on data, where class_column is the column number of target var."""
    def fit(self, data, class_column):
        
        if (not isinstance(data, pd.DataFrame) or class_column not in data.columns):
            raise Exception("Invalid input.")
            
        self.data = data
        self.class_column = class_column
        self.non_class_columns = [c for c in data.columns if c != class_column]

        self.root = self.recursive_build_tree(data, depth=0, name='0')
    

    def recursive_build_tree(self, data, depth, name):
        node = Node()
        node.name = name
        node.data = data

        # base case, when current depth is equal to max depth
        # create a leaf node and return it
        if depth == self.max_depth:
            node.node_type = 'leaf'
            node.label = node.data[self.class_column].mode()[0]
            return node
            
        # otherwise, find the maximum split among columns
        # we iterate through feature columns to find the best column for splitting
        split_column = list(data.columns)
        
        max_info_gain_column = 0
        max_info_gain = -float("inf")
        for i in range(len(split_column)-1): # -1 to exclude target column
            split = Split(data, self.class_column,split_column[i])
            # update max_info_gain and max_info_gain_column
            cur_info_gain = split.info_gain
            if cur_info_gain > max_info_gain:
                max_info_gain = cur_info_gain
                max_info_gain_column = split.split_column
            
        # after finding the max_info_gain, actually do splitting
        split = Split(data, self.class_column, max_info_gain_column)
        node.split = split
        new_data = split.partitions()
        
        # to recursively split on the subtree, we decide to exclude 
        # attributes used by any parent node (although this is not required)
        left_data  = new_data[0].drop(columns=[max_info_gain_column])
        right_data = new_data[1].drop(columns=[max_info_gain_column])
        left_data .reset_index(drop=True,inplace=True)
        right_data.reset_index(drop=True,inplace=True)
        
        # if one child has no examples, stop diverging
        if left_data.empty or right_data.empty:
            node.node_type = 'leaf'
            node.label = node.data[self.class_column].mode()[0]
        else:
            node.children.append(self.recursive_build_tree(left_data , depth+1, name+'.0'))
            node.children.append(self.recursive_build_tree(right_data, depth+1, name+'.1'))
        return node

    
    def predict(self, test):
        # make predictions on the test set
        predictions = []
        for i in range(len(test)):
            cur_data = test.iloc[[i]] # row data at index i 
            pred_y = self.predict_y(self.root, cur_data) # recursively traverse the tree
            predictions.append(pred_y)
        return predictions
    

    def predict_y(self, node, cur_data):
        if node.node_type == 'leaf':
            return node.label
        
        split = node.split
        split_val = split.point
        col = split.split_column
        my_val = cur_data[col] 
        my_val = my_val.iloc[0]
        
        # predict the label based on my_val and split_val
        if my_val <= split_val: 
            return self.predict_y(node.children[0], cur_data)
        else: 
            return self.predict_y(node.children[1], cur_data)
    

    def print(self):
        # Print the current decision tree using DFS order. For
        # each node print whether it is an internal node or a leaf and number
        # of rows from data it received. If it is an internal node, also print
        # the column being tested and the test threshold (E.g., “split 45 at
        # 33.3” says that column 45 is being tested against the threshold
        # 33.3). If it is a leaf node, also print the label (class) that will be
        # predicted for any example reaching it.
        self.recursive_print(self.root)
    

    def recursive_print(self, node):
        print(node)
        for u in node.children:
            self.recursive_print(u)


"""
Test our implementation by plotting prediction accuracy vs. max depth
"""
def validation_curve(df):
    # randomly shuffle the dataset rows 
    data = df.sample(frac=1)
    
    # replace the missing data with mode in that column
    split_column = list(data.columns)
    for i in split_column:
        if sum(data[i].isnull()):
            data[i] = data[i].fillna(data[i].mode())
    
    # divide the examples roughly into three partitions
    x = int(len(data)/3)
    data_1 = data.iloc[   :x   , :]
    data_2 = data.iloc[x  :2*x , :]
    data_3 = data.iloc[2*x:    , :]
    data_set = [data_1, data_2, data_3]

    train_scores_mean, test_scores_mean = [], []
    n = len(split_column) # number of columns

    for max_depth in range(2,17,2):
        train_accs, test_accs = [], []
        
        for i in range(3):
            # use 2/3 data as training set, and 1/3 as test set. 
            # train decision tree with max_depth
            # compute accuracy and append to the list
            tree  = DecisionTree(max_depth)
            train = data_set[(i+1)%3].append(data_set[(i-1)%3], ignore_index=True)
            test  = data_set[i]
            
            tree.fit(train, n-1)
            train_acc= sum(np.array(train.iloc[:,n-1])==np.array(tree.predict(train)))/x/2
            test_acc = sum(np.array(test .iloc[:,n-1])==np.array(tree.predict(test )))/x 
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
        train_scores_mean.append(np.mean(train_accs)) 
        test_scores_mean .append(np.mean(test_accs ))
        

    # plot curves
    param_range = range(2,17,2)
    plt.title("Validation Curve of DecisionTree")
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    


validation_curve(df)