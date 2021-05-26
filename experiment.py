from PolicyNetwork import PolicyNetwork
from Environment import Environment
from util import prep_dataset, fetch_model_name
import numpy as np
import tensorflow as tf

# set seeds for np and tf
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)#tf2.0

epochs = 8

# paths for the KG and QA files
#PQ3H dataset及其对应点FreeBase
path_KB = "./datasets/3H-kb.txt"
path_QA = "./datasets/PQ-3H.txt"

# Experiment Settings
T = 3                 	# To change according to QA type
attention = True		# Use Attention Model or not
perceptron = True    	# Use Perceptron for semantic similary scores

# Prep Data
#  prep_dataset返回
#             KG as network x graph object python里有向 graph化的KG
#             list of ([q_word1, q_word2,...,], e_s, ans)             # e_s should be replaced inside the questions also //e_s是topic entity
KG, dataset = prep_dataset(path_KB, path_QA)
inputs = (KG, dataset, T)

# Run Experiments
print('\n\n*********** Policy Network with Perceptron & Attention ***********')
model_name = fetch_model_name('combined')
policy_network = PolicyNetwork(T, model_name) 											# Model uses attention Layer
train_att_per, val_att_per = policy_network.train(inputs, epochs=epochs)

print('\n\n*********** Policy Network with Perceptron Only ***********')
model_name = fetch_model_name('perceptron')
policy_network = PolicyNetwork(T, model_name)
train_per, val_per = policy_network.train(inputs, epochs=epochs, attention=False)		# Model does not use attention layer