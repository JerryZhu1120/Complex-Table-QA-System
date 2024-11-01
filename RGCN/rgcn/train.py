from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from rgcn.layers.graph import GraphConvolution
from rgcn.layers.input_adj import InputAdj
from rgcn.utils import *
import tensorflow as tf
import pickle as pkl

import os
import sys
import time
import argparse
import numpy as np

split1 = 29205
split2 = 32157


def get_split(y, train_idx, test_idx, validation=True):
    # Make dataset splits
    # np.random.shuffle(train_idx)
    if validation:
        idx_train = train_idx[:split1]
        idx_val = train_idx[split1:32157]
        idx_test = train_idx[32157:37953]  # report final score on validation set for hyperparameter optimization
    else:
        idx_train = train_idx
        idx_val = train_idx  # no validation
        idx_test = test_idx

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)

    y_train[idx_train] = np.array(y[idx_train].todense())
    y_val[idx_val] = np.array(y[idx_val].todense())
    y_test[idx_test] = np.array(y[idx_test].todense())

    return y_train, y_val, y_test, idx_train, idx_val, idx_test


np.random.seed()

ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="aifb",
    help="Dataset string ('aifb', 'mutag', 'bgs', 'am')",
)
ap.add_argument("-e", "--epochs", type=int, default=50, help="Number training epochs")
ap.add_argument("-hd", "--hidden", type=int, default=16, help="Number hidden units")
ap.add_argument("-do", "--dropout", type=float, default=0.0, help="Dropout rate")
ap.add_argument(
    "-b", "--bases", type=int, default=-1, help="Number of bases used (-1: all)"
)
ap.add_argument("-lr", "--learnrate", type=float, default=0.01, help="Learning rate")
ap.add_argument(
    "-l2", "--l2norm", type=float, default=0.0, help="L2 normalization of input weights"
)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument("--validation", dest="validation", action="store_true")
fp.add_argument("--testing", dest="validation", action="store_false")
ap.set_defaults(validation=True)
args = vars(ap.parse_args())
print(args)

# Define parameters
DATASET = args["dataset"]
NB_EPOCH = args["epochs"]
VALIDATION = args["validation"]
LR = args["learnrate"]
L2 = args["l2norm"]
HIDDEN = args["hidden"]
BASES = args["bases"]
DO = args["dropout"]

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(dirname + "/pickle/" + DATASET + ".pickle", "rb") as f:
    data = pkl.load(f)


A = data["A"]
y = data["y"]
train_idx = data["train_idx"]
test_idx = data["test_idx"]


# Get dataset splits
y_train, y_val, y_test, idx_train, idx_val, idx_test = get_split(
    y, train_idx, test_idx, VALIDATION
)
train_mask = sample_mask(idx_train, y.shape[0])

num_nodes = A[0].shape[0]
support = len(A)

# Define empty dummy feature matrix (input is ignored as we set featureless=True)
# In case features are available, define them here and set featureless=False.
X = sp.csr_matrix(A[0].shape)

# Normalize adjacency matrices individually
for i in range(len(A)):
    d = np.array(A[i].sum(1)).flatten()
    d_inv = 1.0 / d
    d_inv[np.isinf(d_inv)] = 0.0
    D_inv = sp.diags(d_inv)
    A[i] = D_inv.dot(A[i]).tocsr()

A_in = [InputAdj(sparse=True) for _ in range(support)]
X_in = Input(shape=(X.shape[1],), sparse=True)

# Define model architecture
H = GraphConvolution(
    HIDDEN,
    support,
    num_bases=BASES,
    featureless=True,
    activation="relu",
    W_regularizer=l2(L2),
)([X_in] + A_in)
H = Dropout(DO)(H)
Y = GraphConvolution(y_train.shape[1], support, num_bases=BASES, activation="softmax")(
    [H] + A_in
)


import theano.tensor as T
import theano


def weighted_cross_entropy(output, target):

    # 将输出张量裁剪到防止 log(0) 出现
    # output = T.clip(output, 1e-7, 1 - 1e-7)
    # 计算交叉熵
    weights = [4177, 5285, 1168, 2535, 18992]
    cross_entropy = -T.sum(target * T.log(output), axis=1)
    # 应用权重
    if weights is not None:
        weights = theano.shared(np.asarray(weights, dtype=output.dtype))
        weighted_cross_entropy = cross_entropy * T.sum(weights * target, axis=1)
    else:
        weighted_cross_entropy = cross_entropy
    return T.mean(weighted_cross_entropy)


# Compile model
model = Model(input=[X_in] + A_in, output=Y)

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(lr=LR),
)

preds = None

# Fit
for epoch in range(1, NB_EPOCH + 1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration
    model.fit(
        [X] + A,
        y_train,
        sample_weight=train_mask,
        batch_size=num_nodes,
        nb_epoch=1,
        shuffle=False,
        verbose=0,
    )

    if epoch % 1 == 0:

        # Predict on full dataset
        preds = model.predict([X] + A, batch_size=num_nodes)

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(
            preds, [y_train, y_val], [idx_train, idx_val]
        )

        print(
            "Epoch: {:04d}".format(epoch),
            "train_loss= {:.4f}".format(train_val_loss[0]),
            "train_acc= {:.4f}".format(train_val_acc[0]),
            "val_loss= {:.4f}".format(train_val_loss[1]),
            "val_acc= {:.4f}".format(train_val_acc[1]),
            "time= {:.4f}".format(time.time() - t),
        )

    else:
        print("Epoch: {:04d}".format(epoch), "time= {:.4f}".format(time.time() - t))
# import h5py

# mode = "validation" if VALIDATION else "testing"
# file_path = "model/" + mode + "/" + DATASET + "_" + str(split) + ".h5"
# model.save(file_path)
new_preds = preds[29205:]
print(len(new_preds))
max_list = []
for each in new_preds:
    max_list.append(np.argmax(each))
with open("preds.pkl", "wb") as f:
    pkl.dump(max_list, f)
# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print(
    "Test set results:",
    "loss= {:.4f}".format(test_loss[0]),
    "accuracy= {:.4f}".format(test_acc[0]),
)
