
import tensorflow as tf
import dgl
import dgl.function as fn
import numpy as np
import os
import datetime
from tqdm import tqdm
import random
from sklearn import metrics
import gc
import logging

logging.basicConfig(level=logging.INFO)

class UpdateLayer(tf.keras.layers.Layer):
    """Class to represent the update function in the message passing operation.
    """
    
    def __init__(self, hidden_state_size=16):
        """Method to initialise the update layer

        Args:
            hidden_state_size (int, optional): The size of the hidden state vectors. Defaults to 16.
        """
    
        super(UpdateLayer, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.GRU_cell = tf.keras.layers.GRUCell(self.hidden_state_size)

    def call(self, h, e):
        """Method to feed inputs through the initialised GRU.

        Args:
            h (tensor): Hidden state vector to use as initial state.
            e (tensor): Edge features used as input.

        Returns:
            tensor: Output of the GRU cell
        """
        
        output, h = self.GRU_cell(e, h)
        return output



class MessageLayer(tf.keras.layers.Layer):
    """
    Class to be used in creating the messages between the nodes.
    """

    def __init__(self, hidden_state_size=16):
        """Method to initilase the message layer

        Args:
            hidden_state_size (int, optional): Size of hidden state vector taken as input to call. Defaults to 16.
        """
        super(MessageLayer, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.dense1 = tf.keras.layers.Dense(self.hidden_state_size * self.hidden_state_size)
        self.dense2 = tf.keras.layers.Dense(self.hidden_state_size)

    def call(self, h, e):
        """Method to feed inputs through the message function as initialised above.

        Args:
            h (tensor): Hidden state vector to be used.
            e (tensor): Edge features to be used.

        Returns:
            tensor: Message to be returned.
        """
        

        A = self.dense1(e)
        A_shape = tf.shape(A)
        A = tf.reshape(A, (A_shape[0], self.hidden_state_size, self.hidden_state_size))
        m = tf.matmul(A, tf.expand_dims(h, 2))
        b = self.dense2(e)
        m = tf.add(tf.squeeze(m), b)
        
        return m




class ReadoutLayer(tf.keras.layers.Layer):
    """
    Class to represent the readout function in the MPNN.
    """
        
        
    def __init__(self, representation_size=16, dense_size=256,
                 reguliser=tf.keras.regularizers.l2,
                 regulise_rate=0.001, dropout_rate=0.2,
                 hidden_readout_layers=1,
                 readout_activation="relu",
                 use_set2set=False,
                 n_iters=8):
        """Method to initialise the readout layer.

        Args:
            representation_size (int, optional): Size of representations. Defaults to 16.
            dense_size (int, optional): Size of dense regression layers. Defaults to 256.
            reguliser (tensorflow regulariser, optional): Regulariser to use. Defaults to tf.keras.regularizers.l2.
            regulise_rate (float, optional): Rate of regularisation. Defaults to 0.001.
            dropout_rate (float, optional): Rate of dropout for ANNs. Defaults to 0.2.
            hidden_readout_layers (int, optional): Hidden readoutlayers to use. Defaults to 1.
            readout_activation (str, optional): What activation to use for the readout. Defaults to "relu".
            use_set2set (bool, optional): Whether to use set2set to encode (more computationally complex). Defaults to False.
            n_iters (int, optional): Iterations to use for set2set representation if used. Defaults to 8.
        """
        super(ReadoutLayer, self).__init__()
        self.use_set2set = use_set2set
        if use_set2set:
            self.input_dim = representation_size
            self.output_dim = 2 * representation_size
            self.n_iters = n_iters
            self.n_layers = 1
            self.lstm = tf.keras.layers.LSTM(self.input_dim, dropout=dropout_rate, return_state=True)
        
        self.hidden_readout_layers = hidden_readout_layers
        if reguliser is not None:
            reguliser = reguliser(regulise_rate)
        self.dense1_i = tf.keras.layers.Dense(representation_size, activation="tanh",
                                              kernel_regularizer=reguliser)
        self.dense1_j = tf.keras.layers.Dense(representation_size,
                                              kernel_regularizer=reguliser,
                                              activation="relu")
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

        if dropout_rate > 0:
            self.inference_layers = [tf.keras.layers.Dense(dense_size, activation=readout_activation,
                                                           kernel_regularizer=reguliser) for i in
                                     range(hidden_readout_layers)]
            self.dropout_layers = [tf.keras.layers.Dropout(rate=dropout_rate) for i in range(hidden_readout_layers)]
            tupled_list = [item for item in zip(self.inference_layers, self.dropout_layers)]
            output_layers = [a for tup in tupled_list for a in tup]
            self.output_layers = tf.keras.Sequential(output_layers)
        else:
            self.inference_layers = [tf.keras.layers.Dense(dense_size, activation=readout_activation,
                                                           kernel_regularizer=reguliser) for i in
                                     range(hidden_readout_layers)]
            self.output_layers = tf.keras.Sequential(self.inference_layers)

        self.regression_layer = tf.keras.layers.Dense(1, activation="linear")

    def RR_calc(self, h, x):
        """Method to calculate RR vector.

        Args:
            h (tensor): Hidden state vector.
            x (tensor): Node features.

        Returns:
            tensor: RR vector
        """
        hx = tf.concat([h, x], axis=1)
        i = self.dense1_i(hx)
        j = self.dense1_j(h)
        RR = tf.nn.sigmoid(i)
        RR = tf.multiply(RR, j)
        del i, j, hx
        return RR

        
        
    def call(self, x, h, graph, training=False):
        """Method to initialise the feed through of the inputs.

        Args:
            x (tensor): Initial node features to be used (|V|*|x_v|)
            h (tensor): Hidden state vector to be used (|V|*|h|)
            graph (DGL_graph): DGL graph to be used.
            training (bool, optional): _description_. Defaults to False.

        Returns:
            tensor (1x): readout value.
        """
        
        RR = self.RR_calc(h, x)
        logging.debug("call output")
        logging.debug(gc.get_stats())
        graph.ndata["rv"] = RR
        if self.use_set2set:
        
            batch_size = graph.batch_size
            q_star = tf.zeros((batch_size, self.output_dim))
            h = tf.zeros((batch_size, self.input_dim))
            c = tf.zeros((batch_size, self.input_dim))
            for _ in range(self.n_iters):
                q,h,c = self.lstm(tf.expand_dims(q_star, axis=1), training=training, initial_state=[h,c])
                # h, c = s[0], s[1]

                q = tf.squeeze(q)
                e = tf.reduce_sum(RR * dgl.broadcast_nodes(graph, q), axis=-1, keepdims=True)

                graph.ndata['e'] = e
                alpha = dgl.softmax_nodes(graph, 'e')
                graph.ndata['r'] = RR * alpha
                readout = dgl.sum_nodes(graph, 'r')
                q_star = tf.concat([q, readout], axis=-1)
            output = q_star
        else:
            output = dgl.sum_nodes(graph, "rv")
        output = self.batch_norm(output, training=training)
        output = self.output_layers(output, training=training)
        output = self.regression_layer(output)
        del RR
        return output


class MPNN(tf.keras.Model):
    """
    Class to build the Message Passing Neural Network.
    """

    def __init__(self, config,
                 optimizer=tf.keras.optimizers.Adam,
                 loss=tf.keras.losses.MeanSquaredError,
                 reguliser=tf.keras.regularizers.l2,
                 T=8, mean=None, std=None,
                 weights_tied=False, 
                 log_path="/scratch/datasets/MPNN"):
        """Initialisation of the MPNN.

        Args:
            config (_type_): _description_
            optimizer (_type_, optional): _description_. Defaults to tf.keras.optimizers.Adam.
            loss (_type_, optional): _description_. Defaults to tf.keras.losses.MeanSquaredError.
            reguliser (_type_, optional): _description_. Defaults to tf.keras.regularizers.l2.
            T (int, optional): _description_. Defaults to 8.
            mean (_type_, optional): _description_. Defaults to None.
            std (_type_, optional): _description_. Defaults to None.
            weights_tied (bool, optional): _description_. Defaults to False.
        """
        super(MPNN, self).__init__()
        self.config = config
        self.hidden_size = config["hidden_vector_size"]
        self.dropout_rate = config["dropout_rate"]
        self.regulise_rate = config["regulariser_rate"]
        self.learning_rate = config["learning_rate"]
        self.hidden_readout_layers = config["hidden_readout_layers"]
        self.hidden_readout_layer_size = config["hidden_readout_layer_size"]
        self.decay_rate = config["learning_rate_decay_rate"]
        self.decay_steps = config["decay_steps"]
        set2set = config["set2set"]
        self.mean = mean
        self.std = std
        self.T = T
        self.reguliser = reguliser
        self.weights_tied = weights_tied
        self.log_path = log_path
        if weights_tied:
            self.message_layer = MessageLayer(hidden_state_size=self.hidden_size)
            self.update_layer = UpdateLayer(hidden_state_size=self.hidden_size)
            self.readout_layer = ReadoutLayer(reguliser=reguliser,
                                                  regulise_rate=self.regulise_rate,
                                                  dropout_rate=self.dropout_rate,
                                                  hidden_readout_layers=self.hidden_readout_layers,
                                                  dense_size=self.hidden_readout_layer_size,
                                                  use_set2set=set2set,
                                                  n_iters=8)
        else:
            self.message_layers = [MessageLayer(hidden_state_size=self.hidden_size) for t in range(T)]
            self.update_layers = [UpdateLayer(hidden_state_size=self.hidden_size) for t in range(T)]
            self.readout_layer = ReadoutLayer(reguliser=reguliser,
                                                  regulise_rate=self.regulise_rate,
                                                  dropout_rate=self.dropout_rate,
                                                  hidden_readout_layers=self.hidden_readout_layers,
                                                  dense_size=self.hidden_readout_layer_size,
                                              use_set2set=set2set,
                                              n_iters=8)



    
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)
        self.dense = tf.keras.layers.Dense(self.hidden_size, activation="relu")
        self.input_feats = tf.keras.layers.Dense(self.hidden_size, activation="relu",
                                                 kernel_regularizer=reguliser(self.regulise_rate))
        self.hidden_feats = tf.keras.layers.Dense(self.hidden_size, activation="relu",
                                                  kernel_regularizer=reguliser(self.regulise_rate))
        self.optimizer = optimizer(learning_rate=lr_schedule)


        self.loss = loss()
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def call(self, graph, training=False):
        """Method to call the MPNN operation with message passing stage.


        Args:
            training (bool, optional): Whether you are training or inferring. Defaults to False.
            T (int, optional): Amount of message passing stages. Defaults to 8.

        Returns:
            returns the inferred output: the regression target in this case.
        """

        with graph.local_scope():
            # Read features and hidden vectors in through a dense layer to make variable structures fixed.
            graph.ndata["feature"] = self.input_feats(graph.ndata["feature"])
            graph.ndata["h"] = self.hidden_feats(graph.ndata["h"])
            logging.debug("assigning h")
            logging.debug(gc.get_stats())

            for t in range(self.T):
                if self.weights_tied:
                    # Pass and aggregate messages
                    graph.update_all(self.message_function, fn.sum("m", "neigh"))
                    logging.debug("message passing")
                    logging.debug(gc.get_stats())
                    # Update the hidden features h
                    out = self.update_layer(graph.ndata["h"], graph.ndata["neigh"])
                    graph.ndata["h"] = out
                else:
                    message_function = lambda edges: {'m': self.message_layers[t](edges.src["h"], edges.data["feature"])}
                    graph.update_all(message_function, fn.sum("m","neigh"))
                    out = self.update_layers[t](graph.ndata["h"], graph.ndata["neigh"])
                    graph.ndata["h"] = out

            # Read the regression output after the message passing
            output = self.readout_layer(graph.ndata["feature"],
                                        graph.ndata["h"],
                                        graph, training=training)
            # print("call output")
            logging.debug(gc.get_stats())
            del out, graph
            return output


    def save_model(self, path="Models", name="MPNN"):
        """Method To save current model.

        Args:
            path (str, optional): The path where the model should be saved. Defaults to "Models".
            name (str, optional): The name of the file under which the model should be saved. Defaults to "MPNN".
        """
        save_path = os.path.join(self.log_path, path, name, self.current_time)
        self.save_weights(filepath=save_path)
        np.savetxt(save_path+"-standardise.txt", [self.mean, self.std])
        f = open(save_path+"-config.txt", "w")
        self.config["T"] = self.T
        f.write(str(self.config))
        f.close()

    def load_model(self, path="Models", name="MPNN"):
        """Method to load model from a path and name.

        Args:
            path (str, optional): Path to the saved model. Defaults to "Models".
            name (str, optional): Filename of the model to load. Defaults to "MPNN".
        """
        load_path = os.path.join(self.log_path, path, name)
        self.load_weights(filepath=load_path)
        self.mean, self.std = np.loadtxt(load_path+"-standardise.txt")

    def setup_logger(self, path="logs", name="MPNN"):
        """Method that sets up the tensorflow logger for use with tensorboard.

        Args:
            path (str, optional): Path where to save logs. Defaults to "logs".
            name (str, optional): Name under which to save it. Defaults to "MPNN".
        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir_train = os.path.join(self.log_path, path, name, current_time, "train")
        self.log_dir_test = os.path.join(self.log_path, path, name, current_time, "test")
        print(self.log_dir_train)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir_train)
        self.test_summary_writer = tf.summary.create_file_writer(self.log_dir_test)


    def message_function(self, edges):
        """Method to return message functions with edge data for DGL interface.

        Args:
            edges (_type_): matrix that includes the rows (edges) and h vectors of each edge.

        Returns:
            _type_: dict with message from the message layer
        """
        return {'m': self.message_layer(edges.src["h"], edges.data["feature"])}


    def train_step(self, data, distribute=False):
        """
        Method to complete a training step.
        :param data: graph, label
        :return: results
        """
        graph_batch, labels = data

        self.graph = graph_batch
        with tf.GradientTape() as tape:
            out = self.call(training=True)
            if distribute == True:
                loss = self.compute_dist_loss(labels, out)
            else:
                loss = self.loss(labels, out)
            if self.reguliser is not None:
                reg_loss = tf.add_n(self.losses)  # getting reg loss
                loss = tf.add(loss, tf.multiply(reg_loss, self.reguliser_rate))
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # self.compiled_metrics.update_state(labels, out)
        return loss



    def train(self, graph_batches, validation_set=None, epoch_len=100, T=8, save_model=False, name=None,
              gpu=0, _ray=False):
        """Method that does the training of the MPNN model.

        Args:
            graph_batches (_type_): _description_
            validation_set (_type_, optional): _description_. Defaults to None.
            epoch_len (int, optional): _description_. Defaults to 100.
            T (int, optional): _description_. Defaults to 8.
            save_model (bool, optional): _description_. Defaults to False.
            name (_type_, optional): _description_. Defaults to None.
            gpu (int, optional): _description_. Defaults to 0.
            _ray (bool, optional): _description_. Defaults to False.
        """

        loss = 0
        R_2 = 0

        self.setup_logger(name=name)
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        val_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        performance = [0]
        tf.keras.backend.clear_session()

        for epoch in range(epoch_len):
            epoch_loss = []
            r_2_list = []
            

            random.shuffle(graph_batches)
            batch_loss_avg = []
            for ind, (graph_batch, labels) in enumerate(tqdm(graph_batches, desc="epoch: {}".format(epoch))):
                with tf.GradientTape() as tape:
                    out = self.call(graph_batch, training=True)
                    loss = self.loss(labels, out)
                    
                    logging.debug("loss: {} type: {}".format(loss, type(loss)))

                    if self.reguliser is not None:
                        reg_loss = tf.add_n(self.losses)  # getting reg loss
                        loss = tf.add(loss, tf.multiply(reg_loss, self.regulise_rate))
                    
                    
                    logging.debug("loss with reguliser: {} type: {}".format(loss, type(loss)))
            
                    # Updating metric for the tensorboard tracker
                    train_loss(loss)
                    
                    grads = tape.gradient(loss, self.trainable_weights)
                    logging.debug("grads: {} type: {}".format(grads, type(grads)))
                    logging.debug("applying gradients")

                    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                    r_2_list.append(metrics.r2_score(labels.numpy(), tf.squeeze(out).numpy()))
                    
                del loss, out, grads, reg_loss
            self.save_model(name=name)
            gc.collect()


            with self.train_summary_writer.as_default():
                logging.info("training loss: {}".format(round(train_loss.result().numpy(),3)))
                logging.info("average batch R2: {}".format(round(np.mean(r_2_list),3)))
                
                tf.summary.scalar("R2", np.mean(r_2_list), step=epoch)
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
            

            if validation_set is not None:
                R_2, val_loss = self.evaluate(validation_set)
                logging.info("Validation R2: {}".format(round(R_2,3)))
                if save_model == 1:
    
                    if val_loss <= min(performance):
                        self.save_model(name=name)
                        logging.info("saved model")

                    performance.append(val_loss)
                with self.test_summary_writer.as_default():
                    
                    tf.summary.scalar('R2', R_2, step=epoch)
                    tf.summary.scalar('loss', val_loss, step=epoch)
                

    def evaluate(self, data):
        """Method to evaluate a set of data - consisting of graphs and labels

        Args:
            data (list of tuples: (graph, labels)): each item in the list is a graph with its corresponding property to regress on.

        Returns:
            tuple(float, float): Coefficient of determination R^2 and the loss
        """
        for graph, labels in data:
            
            out = self.call(graph)
            out = tf.squeeze(out).numpy()
            labels = tf.squeeze(labels).numpy()
        
            R_2 = metrics.r2_score(labels, out)
            loss = self.loss(labels, out)
        return R_2, loss

    def infer(self, graph):
        """Method to infer a property from a graph.

        Args:
            graph (dgl_graph): either a dgl graph or a batch of dgl graphs

        Returns:
            float: graph property that is infered
        """
        out = self.call(graph)
        out = tf.squeeze(out).numpy()
        out *= self.std
        out += self.mean
        return out


