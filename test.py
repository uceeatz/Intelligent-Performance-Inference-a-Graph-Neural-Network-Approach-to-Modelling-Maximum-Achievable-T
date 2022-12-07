import tensorflow as tf
from mpnn_model import MPNN
import scipy
from sklearn import metrics
import time
import numpy as np
from tqdm import tqdm
import os
import data


# Terminal arguments
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


# Cofiguration of the MPNN model
config = {
            "hidden_vector_size": 16,
            "learning_rate": 1,
            "batch_size": 50,
            "dropout_rate": 1,
            "regulariser_rate": 1,
            "hidden_readout_layers": 1,
            "hidden_readout_layer_size": 256,
            "learning_rate_decay_rate": 0.96,
            "set2set":0,
            "decay_steps": 20000,
            "name": "MPNN",
            "save_model": 1,
            "epochs":20000
        }

# 10-15 nodes: "MPNN-weights-tied-15-T/20210627-231759"
# 25-45 nodes: "MPNN-weights-tied-45-T/20210628-175348"
# 55-100 nodes: "MPNN-weights-tied-100-T/20210706-095852"
parameters = {"nodes":[10,15],
              "label":"ILP-connections Capacity",
              "node features": ["degree", "traffic"],
              "edge features": ["worst case NSR"],
              "mpnn data label": "MPNN Capacity",
              "model name":"MPNN-weights-tied-45-T/20210628-175348",
              "filepath":"/rdata/ong/robin/MPNN/hdf5/MPNN-uniform-25-45-test.bin"
              }



# Initialising the capacity regressor from the MPNN model
capacity_regressor = MPNN(config, T=8, weights_tied=True, log_path="/scratch/datasets/MPNN")
capacity_regressor.load_model(name=parameters["model name"])



# Reading and batching data
data_test, mean, std = data.load_batch_data(parameters["filepath"],
                                            batch_size=1,
                                            standardise=False,
                                            node_features=parameters["node features"],
                                            edge_features=parameters["edge features"])





labels_list = []
throughput_pred_list = []
time_taken_list = []
R_2 = 0
for graph, label in tqdm(data_test, desc="Running testing"):
    # start time
    time_start = time.perf_counter()
    # throughput predict from capacity regressor
    throughput_pred = capacity_regressor.infer(graph)
    # stop time
    time_taken = time.perf_counter() - time_start
    
    time_taken_list.append(time_taken)
    labels_list.append(tf.squeeze(label).numpy())
    throughput_pred_list.append(throughput_pred)
    
    

R_2 = metrics.r2_score(labels_list, throughput_pred_list)
pearson_corr = scipy.stats.pearsonr(labels_list, throughput_pred_list)
avg_time_taken = np.array(time_taken_list).mean()

print("R2: {}".format(R_2))
print("p: {}".format(pearson_corr))
print("average time taken: {}ms".format(round(avg_time_taken*1e3,2)))


