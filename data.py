import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    student_num: int
    exercise_num: int
    knowledge_num: int
    q_matrix: np.ndarray
    response: np.ndarray
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def _read_csv(path, dtype):
    return pd.read_csv(path, header=None).to_numpy(dtype=dtype)


def load_dataset(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    info = config.get("info", config)
    return DatasetBundle(
        student_num=int(info["student_num"]),
        exercise_num=int(info["exercise_num"]),
        knowledge_num=int(info["knowledge_num"]),
        q_matrix=_read_csv(root / "q_matrix.csv", np.float32),
        response=_read_csv(root / "response.csv", np.int64),
        train=_read_csv(root / "train.csv", np.int64),
        valid=_read_csv(root / "val.csv", np.int64),
        test=_read_csv(root / "test.csv", np.int64),
    )


class ResponseDataset(Dataset):
    def __init__(self, rows, q_matrix):
        self.rows = torch.as_tensor(rows, dtype=torch.long)
        self.q_matrix = torch.as_tensor(q_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        student, item, label = self.rows[index]
        return student, item, self.q_matrix[item], label.float()


def _normalize(matrix):
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(degree)
    nonzero = degree > 0
    inverse[nonzero] = degree[nonzero] ** -0.5
    scale = sp.diags(inverse)
    return scale.dot(matrix).dot(scale).tocsr()


def _to_sparse_tensor(matrix):
    coo = matrix.tocoo().astype(np.float64)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def _binary_csr(rows, cols, shape):
    matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=shape,
    )
    if matrix.nnz:
        matrix.data[:] = 1.0
    return matrix


def _response_graph(rows, student_num, exercise_num, self_loops):
    se = _binary_csr(rows[:, 0], rows[:, 1], (student_num, exercise_num))
    graph = sp.bmat(
        [
            [sp.csr_matrix((student_num, student_num)), se],
            [se.T, sp.csr_matrix((exercise_num, exercise_num))],
        ],
        format="csr",
        dtype=np.float64,
    )
    if self_loops:
        graph = graph + sp.eye(graph.shape[0], format="csr", dtype=np.float64)
    return _to_sparse_tensor(_normalize(graph))


def _evidence(rows, q_matrix, student_num, knowledge_num):
    if len(rows) == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.zeros(student_num, knowledge_num),
        )
    q_rows = sp.csr_matrix(q_matrix)[rows[:, 1]].tocoo()
    edge_student = rows[q_rows.row, 0].astype(np.int64, copy=False)
    edge_item = rows[q_rows.row, 1].astype(np.int64, copy=False)
    edge_concept = q_rows.col.astype(np.int64, copy=False)
    count = np.zeros(student_num * knowledge_num, dtype=np.float32)
    np.add.at(count, edge_student * knowledge_num + edge_concept, 1.0)
    return (
        torch.from_numpy(edge_student),
        torch.from_numpy(edge_item),
        torch.from_numpy(edge_concept),
        torch.from_numpy(count.reshape(student_num, knowledge_num)),
    )


def build_graphs(response, q_matrix, student_num, exercise_num, response_self_loops=True):
    response = np.asarray(response, dtype=np.int64)
    q_matrix = np.asarray(q_matrix, dtype=np.float32)
    knowledge_num = q_matrix.shape[1]
    q = sp.csr_matrix(q_matrix, dtype=np.float64)
    q_graph = sp.bmat(
        [
            [sp.csr_matrix((exercise_num, exercise_num)), q],
            [q.T, sp.csr_matrix((knowledge_num, knowledge_num))],
        ],
        format="csr",
        dtype=np.float64,
    )
    positive_rows = response[response[:, 2] == 1]
    negative_rows = response[response[:, 2] == 0]
    graphs = {
        "q_graph": _to_sparse_tensor(_normalize(q_graph)),
        "right_response_graph": _response_graph(
            positive_rows, student_num, exercise_num, response_self_loops
        ),
        "wrong_response_graph": _response_graph(
            negative_rows, student_num, exercise_num, response_self_loops
        ),
    }
    for prefix, rows in (("positive", positive_rows), ("negative", negative_rows)):
        students, items, concepts, count = _evidence(
            rows, q_matrix, student_num, knowledge_num
        )
        graphs[f"{prefix}_edge_student"] = students
        graphs[f"{prefix}_edge_item"] = items
        graphs[f"{prefix}_edge_concept"] = concepts
        graphs[f"concept_{prefix}_count"] = count
    return graphs
