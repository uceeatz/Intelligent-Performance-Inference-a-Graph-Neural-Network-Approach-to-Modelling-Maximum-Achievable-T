
import tensorflow as tf
from mpnn_model import MPNN
import argparse
import os
import logging
import data

logging.basicConfig(level=logging.INFO)




os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
print("set environ")
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

parser = argparse.ArgumentParser(description='Train MPNN model.')
parser.add_argument('-vc', action='store', type=int, default=200, help="Maximum amount of data to assign for validation")
parser.add_argument('-lrd', action='store', type=float, default=0.96, help='learning rate decay rate - exponential decay')
parser.add_argument('-lrds', action='store', type=float, default=10000, help='learning rate decay steps - exponential decay')
parser.add_argument('-s', action='store', type=int, default=0, help="Whether to save the model or not")
parser.add_argument('--name', action='store', type=str, default="MPNN", help="Name used to save the model")
parser.add_argument('--batch', action='store', type=int, default=50, help="How large a batch to use for training")
parser.add_argument('-r', action='store', type=str, default=None, help="what type of regularisation")
parser.add_argument('-rr', action='store', type=float, default=0.001, help="regularisation rate")
parser.add_argument('-dr', action='store', type=float, default=0.2, help="dropout rate")
parser.add_argument('-lr', action='store', type=float, default=0.2, help="learning rate")
parser.add_argument('--gpu', action='store', type=int, default=0, help="which gpu to use")
parser.add_argument('-rs', action='store', type=int, default=256, help="size of readout layer")
parser.add_argument('-rl', action='store', type=int, default=1, help="amount of readout layers")
parser.add_argument('-hvs', action='store', type=int, default=16, help="size of hidden vector")
parser.add_argument('-s2s', action='store', type=int, default=0, help="whether to use set2set or not")
parser.add_argument('-t', action='store', type=int, default=8, help="how many message passing stages to use")
args = parser.parse_args()




model_data = {"nodes":[10,15],
              "label":"ILP-connections Capacity",
              "node features" : ["degree", "traffic"],
              "edge features" : ["worst case NSR"],
              "filename":"MPNN-uniform.bin",
              "path":'/rdata/ong/robin/MPNN/hdf5'
              }
logging.info("reading data in batches of {}".format(args.batch))
batched_data, mean, std = data.load_batch_data("{}/{}".format(model_data["path"], model_data["filename"]),
                                            batch_size=args.batch, standardise=True, node_features=model_data["node features"],
                                            edge_features=model_data["edge features"])

train_data = batched_data[:-int(args.vc/args.batch)]
val_data = batched_data[-int(args.vc/args.batch):]

config = {
    "hidden_vector_size": args.hvs,
    "learning_rate": args.lr,
    "batch_size": args.batch,
    "dropout_rate": args.dr,
    "regulariser_rate": args.rr,
    "hidden_readout_layers": args.rl,
    "hidden_readout_layer_size": args.rs,
    "learning_rate_decay_rate": args.lrd,
    "decay_steps": args.lrds,
    "name": args.name,
    "save_model": 1,
    "epochs": 2000,
    "node features":model_data["node features"],
    "edge features":model_data["edge features"],
    "training data path": model_data["path"],
    "training data filename": model_data["filename"],
    "training labels": model_data["label"],
    "training nodes": model_data["nodes"],
    "message passing iterations":args.t,
    "set2set":args.s2s,
    "regularisation type": args.r,
    "val data count":args.vc,
    "training data count":args.batch*len(train_data),
}
logging.info("read data -> setting up MPNN model")
capacity_regressor = MPNN(config, mean=mean, std=std, weights_tied=True, T=args.t)
logging.info("starting training process...")
capacity_regressor.train(train_data, validation_set=val_data, epoch_len=2000, save_model=args.s, name=args.name,
                            gpu=args.gpu)



