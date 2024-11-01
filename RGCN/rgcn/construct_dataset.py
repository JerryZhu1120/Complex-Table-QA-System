import json
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from rgcn.utils import *
import pickle as pkl
import time

count = []


def load_table_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for each in data:
        nodes = []
        for i in each["cell_ID_matrix"]:
            for j in i:
                if j not in nodes:
                    nodes.append(j)
        count.append(len(nodes))
    total = 0
    for each in count:
        total += each
    down = []
    right = []
    adj_shape = (total, total)
    print(total)
    for i, each in enumerate(data):
        offset = sum(count[:i])
        # print("offset:", offset)
        for j, row in enumerate(each["cell_ID_matrix"]):
            for k, cell in enumerate(row):
                if j + 1 < len(each["cell_ID_matrix"]):
                    if cell != each["cell_ID_matrix"][j + 1][k]:
                        down.append(
                            (cell + offset, each["cell_ID_matrix"][j + 1][k] + offset)
                        )
                if k + 1 < len(each["cell_ID_matrix"][0]):
                    if cell != each["cell_ID_matrix"][j][k + 1]:
                        right.append(
                            (cell + offset, each["cell_ID_matrix"][j][k + 1] + offset)
                        )
    down_row = [each[0] for each in down]
    down_col = [each[1] for each in down]
    right_row = [each[0] for each in right]
    right_col = [each[1] for each in right]
    meta_data = np.ones(len(down_row), dtype=np.int8)
    down_matrix = sp.csr_matrix(
        (meta_data, (down_row, down_col)), shape=adj_shape, dtype=np.int8
    )
    up_matrix = sp.csr_matrix(
        (meta_data, (down_col, down_row)), shape=adj_shape, dtype=np.int8
    )
    meta_data = np.ones(len(right_row), dtype=np.int8)
    right_matrix = sp.csr_matrix(
        (meta_data, (right_row, right_col)), shape=adj_shape, dtype=np.int8
    )
    left_matrix = sp.csr_matrix(
        (meta_data, (right_col, right_row)), shape=adj_shape, dtype=np.int8
    )
    A = [down_matrix, up_matrix, right_matrix, left_matrix]
    A.append(sp.identity(A[0].shape[0]).tocsr())
    labeled_nodes_idx = []
    labels = sp.lil_matrix((adj_shape[0], 5))
    for i, each in enumerate(data):
        offset = sum(count[:i])
        if "column_attribute" not in each:
            continue
        for x in range(count[i]):
            if x in each["column_attribute"]:
                labeled_nodes_idx.append(x + offset)
                labels[x + offset, 0] = 1
            elif x in each["row_attribute"]:
                labeled_nodes_idx.append(x + offset)
                labels[x + offset, 1] = 1
            elif x in each["column_index"]:
                labeled_nodes_idx.append(x + offset)
                labels[x + offset, 2] = 1
            elif x in each["row_index"]:
                labeled_nodes_idx.append(x + offset)
                labels[x + offset, 3] = 1
            else:
                labeled_nodes_idx.append(x + offset)
                labels[x + offset, 4] = 1
    # print(labeled_nodes_idx)
    # print(labels)
    print("Calculating level sets...")
    t = time.time()
    # Get level sets (used for memory optimization)
    bfs_generator = bfs_relational(A, labeled_nodes_idx)
    lvls = list()
    lvls.append(set(labeled_nodes_idx))
    lvls.append(set.union(*next(bfs_generator)))
    print("Done! Elapsed time " + str(time.time() - t))

    # Delete unnecessary rows in adjacencies for memory efficiency
    todel = list(set(range(total)) - set.union(lvls[0], lvls[1]))
    for i in range(len(A)):
        csr_zero_rows(A[i], todel)
    return A, labels


A, labels = load_table_data("../../table_data/train_dev_test_new.json")
train_idx = list(range(A[0].shape[0]))
print(len(train_idx))
test_idx = []
data = {"A": A, "y": labels, "train_idx": train_idx, "test_idx": test_idx}

# with open("preds_6788.pkl", "rb") as f:
#     labels = pkl.load(f)
# print(len(labels))
# with open("../../table_data/test_data.json", "r") as f:
#     json_data = json.load(f)
# for i, each in enumerate(json_data):
#     offset = sum(count[: i + 936]) - sum(count[:936])
#     print(offset)
#     each["column_attribute"] = []
#     each["row_attribute"] = []
#     each["column_index"] = []
#     each["row_index"] = []
#     for j in range(count[i + 936]):
#         if labels[j + offset] == 0:
#             each["column_attribute"].append(j)
#         if labels[j + offset] == 1:
#             each["row_attribute"].append(j)
#         if labels[j + offset] == 2:
#             each["column_index"].append(j)
#         if labels[j + offset] == 3:
#             each["row_index"].append(j)
# with open("../../table_data/test_tables_labeled.json", "w") as f:
#     json.dump(json_data, f, indent=4, ensure_ascii=False)
with open("pickle/test_table.pickle", "wb") as f:
    pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
