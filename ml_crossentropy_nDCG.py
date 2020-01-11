import numpy as np
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

 #Function to load data
def load_LETOR4(file, num_features=46):
	'''
	:param file: the input file
	:param num_features: the number of features
	:return: the list of tuples, each tuple consists of qid, doc_reprs, doc_labels
	'''
  
	feature_cols = [str(f_index) for f_index in range(1, num_features + 1)]

	df = pd.read_csv(file, sep=" ", header=None)
	df.drop(columns=df.columns[[-2, -3, -5, -6, -8, -9]], axis=1, inplace=True)  # remove redundant keys
	assert num_features == len(df.columns) - 5

	for c in range(1, num_features +2):           							 # remove keys per column from key:value
		df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])

	df.columns = ['rele_truth', 'qid'] + feature_cols + ['#docid', 'inc', 'prob']

	for c in ['rele_truth'] + feature_cols:
		df[c] = df[c].astype(np.float32)

	df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)  # additional binarized column for later filtering

	list_Qs = []
	qids = df.qid.unique()
	np.random.shuffle(qids)
	for qid in qids:
		sorted_qdf = df[df.qid == qid].sort_values('rele_truth', ascending=False)

		doc_reprs = sorted_qdf[feature_cols].values
		doc_labels = sorted_qdf['rele_truth'].values

		list_Qs.append((qid, doc_reprs, doc_labels))

	#if buffer: pickle_save(list_Qs, file=perquery_file)

	return list_Qs

  # Function to compute nDCG score
def discounted_cumu_gain_at_k(sorted_labels, cutoff):
	'''
	:param sorted_labels: ranked labels (either standard or predicted by a system) in the form of np array
	:param max_cutoff: the maximum rank position to be considered
	:param multi_lavel_rele: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
	:return: cumulative gains for each rank position
	'''
	nums = np.power(2.0, sorted_labels[0:cutoff]) - 1.0

	denoms = np.log2(np.arange(cutoff) + 2.0)  # discounting factor
	dited_cumu_gain = np.sum(nums / denoms)

	return dited_cumu_gain

def ndcg_at_k(sys_sorted_labels, ideal_sorted_labels, k):
	sys_discounted_cumu_gain_at_k = discounted_cumu_gain_at_k(sys_sorted_labels, cutoff=k)
	ideal_discounted_cumu_gain_at_k = discounted_cumu_gain_at_k(ideal_sorted_labels, cutoff=k)
	ndcg_at_k = sys_discounted_cumu_gain_at_k / ideal_discounted_cumu_gain_at_k
	return ndcg_at_k

  # Function to find the sigmoid of an x value
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

    # Function to find the cross entropy value from the definiton of cross entropy
# prediction: f(x) value
# y: standard label
def cross_entropy(prediction, y):
  return -y * np.log(prediction) - (1 - y) * np.log(1 - prediction)

  # Function to apply gradient descent method on dataset
# list_Qs: feature values and initial predicted values of training dataset
# theta: model parameters
# learning_rate: how much a prediction differs from the previous prediction while moving towards the minimum of loss function
# repeat_times: how many times the gradient descent method will iterate
def cla_gradient_descent(list_Qs, theta, learning_rate=0.001, repeat_times=5):
  # cost_history is an array with the size of repeat_times, because there will be a new cost at each iteration
  cost_history = np.zeros(repeat_times)

  for k in range(repeat_times):
    for (qid, train_X, train_Y) in list_Qs:
      m = len(train_Y)
      for i in range(m):
        x = train_X[i, :]
        y = train_Y[i]
        # the prediction using sigmoid
        prediction = sigmoid(np.dot(x, theta))
        # new parameter value is the difference between the previous parameter value and learning_rate * the gradient of cost function, 
        # which is feature values * (dot product of feature values and model parameters - predicted value)
        theta = theta - learning_rate*(x*(prediction - y))
    
    # using cross entropy as loss function
    cost=0
    for (qid, train_X, train_Y) in list_Qs:
      predictions_per_query = sigmoid(train_X.dot(theta))
      cost_per_query = 1 / m * np.sum(cross_entropy(predictions_per_query, train_Y))
      cost += cost_per_query
    cost_history[k]  = cost
  return theta, cost_history

  # Function to evaluate the test
def evaluate(test_list_Qs, optimized_theta, k=5):
  nDCG=0.0
  count = 0.0 # count the number of test queries
  for (qid, test_X, test_Y) in test_list_Qs:
    sum_per_query = np.sum(test_Y)
    m = len(test_Y)
    if m < k or sum_per_query <= 0: # filter the queries that 
      continue
    else:
      count += 1
    
    predictions_per_query = test_X.dot(optimized_theta) # the predictions with respect to one query

    ideal_sorted_labels = np.sort(test_Y) # the default is ascending order
    ideal_sorted_labels = np.flip(ideal_sorted_labels) # convert to the descending order
    #print('ideal_sorted_labels', ideal_sorted_labels)

    sorted_pred_indice = np.argsort(-predictions_per_query) # get the indice that sort the predictions in a descending order
    sys_sorted_labels = test_Y[sorted_pred_indice] # get the corresponding ranking of standard labels 
    #print('sys_sorted_labels', sys_sorted_labels)

    nDCG_per_query = ndcg_at_k(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, k=k)
    nDCG += nDCG_per_query

  nDCG = nDCG/count # using the average nDCG
  return nDCG

  # Demonstration
repeat_times = 10
X_Dimension = 46

debug = True # print some information if needed

initial_theta = np.random.randn(X_Dimension) # initialization of the prameters
if debug:
  print('Random initialization of parameters: {}'.format(initial_theta))

# load the training data
file = '/content/vali_as_train.txt'
train_list_Qs = load_LETOR4(file=file)

# training the model
optimized_theta, cost_history = cla_gradient_descent(train_list_Qs, initial_theta, learning_rate=0.001, repeat_times=repeat_times)

if debug:
  print('\n Optimized parameters:{}'.format(optimized_theta))

# evaluate the ranking model by computing its nDCG score
# load the test data
file = '/content/test.txt'
test_list_Qs = load_LETOR4(file=file)

nDCG = evaluate(test_list_Qs=test_list_Qs, optimized_theta=optimized_theta)
print('\n The nDCG score of the optimized ranking model is:', nDCG)


# show the cost variation w.r.t. the training process
print()
fig, ax = plt.subplots(figsize=(15,8))
ax.set_ylabel('Cost')
ax.set_xlabel('Number of times')
ax.plot(range(repeat_times), cost_history[:repeat_times], 'b.')


