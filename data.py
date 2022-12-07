
import dgl
import tensorflow as tf
import random
from tqdm import tqdm
from dgl.data.utils import load_graphs
import logging


logging.basicConfig(level=logging.INFO)


def load_batch_data(path, batch_size=50, standardise=False, mean=None, std=None, node_features=None, edge_features=None, hidden_shape_size=16):
    """Method to load batched data from a dgl .bin file. 

    Args:
        path (str): Path to load data from including the datafile (.bin).
        batch_size (int, optional): Size of batch to use. Defaults to 50.
        standardise (bool, optional): Whether to standardise the labels. Defaults to False.
        mean (float, optional): Which mean to use to standardise against. Defaults to None.
        std (float, optional): Standard deviation to standardise against. Defaults to None.
        node_features (list of str, optional): Node features to use. Defaults to None.
        edge_features (list of str, optional): Edge features to use. Defaults to None.
        hidden_shape_size (int, optional): Hidden shape size. Defaults to 16.

    Returns:
        (list(tuple), float, float): Tuple of data_list [(graph, label), ...], mean, standard deviation.
    """
    data_list = []
    glist, label_dict = load_graphs(path)
    capacity = label_dict["glabels"].numpy()
    logging.info("data length: {}".format(len(capacity)))
    if node_features is not None and edge_features is not None:
        for graph in tqdm(glist, desc="setting node features: {} and edge features: {}".format(node_features, edge_features)):
            set_features(graph, node_features, edge_features, hidden_shape_size)
    if standardise and (mean ==None and std == None):
        mean = capacity.mean()
        std = capacity.std()
        capacity -= mean
        capacity /= std
    
    indeces = list(range(0,len(glist)))
    batch_ind = list(range(0, len(glist)+1, batch_size))
    random.shuffle(indeces)
    indeces = [indeces[batch_ind[i-1]:batch_ind[i]] for i in range(1, len(batch_ind))]
    for index in tqdm(indeces, desc="batching data from: {}".format(path)):
        graphs = [glist[ind].to("gpu:0") for ind in index]
        capacity_batch = capacity[index]
        batched_graphs = dgl.batch(graphs)
        data_list.append((batched_graphs,tf.convert_to_tensor(capacity_batch)))
    return data_list, mean, std


def set_features(dgl_graph, node_attributes, edge_attributes, hidden_shape_size):
    """Method that sets the features of the dgl graphs.

    Args:
        dgl_graph (dgl_graph): dgl graph to set features on.
        node_attributes (list of str): Node features to write to node feature vector.
        edge_attributes (list of str): Edge features to write to edge feature vector.
        hidden_shape_size (int): Size of hidden vector.
    """
    dgl_graph = dgl_graph.to('gpu:0')
    
    for edge_att in edge_attributes:
        dgl_graph.edata[edge_att] = tf.cast(dgl_graph.edata[edge_att], dtype=tf.float32)
    for node_att in node_attributes:
        dgl_graph.ndata[node_att] = tf.cast(dgl_graph.ndata[node_att], dtype=tf.float32)

    features = [tf.cast(tf.expand_dims(dgl_graph.ndata[node_att], axis=1), dtype=tf.float32) if
                len(dgl_graph.ndata[node_att].shape) == 1 else tf.cast(dgl_graph.ndata[node_att], dtype=tf.float32)
                for node_att in node_attributes]
    feature_vector = tf.concat(features, 1)
    spare_length = hidden_shape_size - feature_vector.shape[1]
    if spare_length > 0:

        hidden_state = tf.concat([tf.cast(feature_vector, dtype=tf.float32),
                                  tf.zeros(tf.shape(len(dgl_graph.nodes().numpy()), spare_length), dtype=tf.float32)],
                                 axis=1)
    else:
        hidden_state = feature_vector
    edges_features = [tf.cast(dgl_graph.edata[edge_att], dtype=tf.float32) for edge_att in edge_attributes]
    edge_feature_vector = tf.stack(edges_features, axis=1)
    

    for node_att in node_attributes:
        dgl_graph.ndata[node_att] = hidden_state
    
    dgl_graph.ndata["feature"] = hidden_state  # changed for varying SNR vector lengths
    dgl_graph.ndata["h"] = hidden_state
    dgl_graph.edata["feature"] = edge_feature_vector
    

