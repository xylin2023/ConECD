import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score


def prediction_metrics(labels, predictions):
    labels = np.asarray(labels).astype(int, copy=False)
    predictions = np.asarray(predictions, dtype=float)
    binary_predictions = (predictions >= 0.5).astype(int)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(labels, predictions))),
        "AUC": float(roc_auc_score(labels, predictions)),
        "ACC": float(accuracy_score(labels, binary_predictions)),
        "F1": float(f1_score(labels, binary_predictions, pos_label=1, zero_division=0)),
    }


def _binary_krc(pairs):
    pairs = sorted(pairs, key=lambda x: x[0])
    rank_sum = sum(i + 1 for i, (_, answer) in enumerate(pairs) if answer == 1)
    positives = sum(answer == 1 for _, answer in pairs)
    negatives = sum(answer == 0 for _, answer in pairs)
    if positives == 0:
        return 0.0
    if negatives == 0:
        return 1.0
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def krc(mastery, responses, q_matrix):
    rows = np.asarray(responses, dtype=np.int64)
    q = np.asarray(q_matrix)
    values = []
    for concept in range(q.shape[1]):
        mask = q[rows[:, 1], concept] == 1
        if not np.any(mask):
            continue
        pairs = [
            (round(float(mastery[s, concept]), 5), int(y))
            for s, y in zip(rows[mask, 0], rows[mask, 2])
        ]
        values.append(_binary_krc(pairs))
    return float(np.mean(values)) if values else np.nan


def empirical_behavior(responses, q_matrix, student_num, knowledge_num):
    rows = np.asarray(responses, dtype=np.int64)
    q = np.asarray(q_matrix)
    sums = np.zeros((student_num, knowledge_num), dtype=np.float32)
    counts = np.zeros((student_num, knowledge_num), dtype=np.int32)
    q_rows, concepts = np.nonzero(q[rows[:, 1]])
    flat = rows[q_rows, 0] * knowledge_num + concepts
    np.add.at(sums.ravel(), flat, rows[q_rows, 2])
    np.add.at(counts.ravel(), flat, 1)
    behavior = np.divide(
        sums,
        counts + 1e-8,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    return behavior, counts


def dca_eca(mastery, behavior, observed, tau=0.5, gamma=0.2):
    counts = observed if np.issubdtype(np.asarray(observed).dtype, np.integer) else np.asarray(observed, dtype=np.int32)
    total = agree = 0
    eca_sum = 0.0
    for student in range(min(len(mastery), len(behavior))):
        concepts = np.flatnonzero(counts[student] > 0)
        if len(concepts) < 2:
            continue
        b = behavior[student, concepts].astype(np.float64, copy=False)
        predicted = mastery[student, concepts].astype(np.float64, copy=False)
        upper = np.triu_indices(len(concepts), 1)
        delta_b = b[upper[0]] - b[upper[1]]
        clear = np.abs(delta_b) >= tau
        delta_m = predicted[upper[0]] - predicted[upper[1]]
        signed = np.sign(delta_b[clear]) * delta_m[clear]
        total += len(signed)
        agree += int(np.sum(signed > 0))
        eca_sum += float(np.clip(signed / gamma, 0, 1).sum())
    return {
        "dca": float(agree / total) if total else np.nan,
        "eca": float(eca_sum / total) if total else np.nan,
        "pairs": int(total),
    }


def _distance(count, sum_x, sum_y, sum_x2, sum_y2, dot):
    safe = np.maximum(count, 1.0)
    euclid = np.sqrt(np.maximum(sum_x2 + sum_y2 - 2 * dot, 0) / safe)
    var_x = np.maximum(sum_x2 - sum_x * sum_x / safe, 0)
    var_y = np.maximum(sum_y2 - sum_y * sum_y / safe, 0)
    denom = np.sqrt(var_x * var_y)
    corr = np.empty_like(euclid, dtype=np.float32)
    valid = denom > 1e-8
    corr[valid] = np.clip(
        (dot[valid] - sum_x[valid] * sum_y[valid] / safe[valid]) / denom[valid],
        -1,
        1,
    )
    corr[~valid] = np.where(euclid[~valid] <= 1e-8, 1.0, -1.0)
    return euclid + (1.0 - corr) / 2.0


def epa(mastery, behavior, counts, block_size=256):
    valid = (counts > 0).astype(np.float32)
    b = behavior.astype(np.float32) * valid
    m = mastery.astype(np.float32) * valid
    n = len(b)
    columns = np.arange(n)
    total_error = 0.0
    pair_count = 0
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        rows = np.arange(start, stop)
        va, ba, ma = valid[start:stop], b[start:stop], m[start:stop]
        common = va @ valid.T
        usable = (common > 0) & (columns[None, :] > rows[:, None])
        if not np.any(usable):
            continue
        db = _distance(
            common, ba @ valid.T, va @ b.T, (ba * ba) @ valid.T,
            va @ (b * b).T, ba @ b.T,
        )
        dm = _distance(
            common, ma @ valid.T, va @ m.T, (ma * ma) @ valid.T,
            va @ (m * m).T, ma @ m.T,
        )
        total_error += float(np.abs(dm[usable] - db[usable]).sum())
        pair_count += int(usable.sum())
    return {
        "epa": float(1.0 - total_error / pair_count / 2.0) if pair_count else np.nan,
        "pairs": pair_count,
    }
