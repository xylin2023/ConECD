import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConECDCore(nn.Module):
    def __init__(
        self,
        student_num,
        exercise_num,
        knowledge_num,
        latent_dim=32,
        q_layers=1,
        response_layers=2,
        eps=1e-8,
        init_gap_scale=1.0,
    ):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.q_layers = q_layers
        self.response_layers = response_layers
        self.eps = eps
        self.graphs = {}

        self.student_emb = nn.Embedding(student_num, latent_dim)
        self.item_emb = nn.Embedding(exercise_num, latent_dim)
        self.concept_emb = nn.Embedding(knowledge_num, latent_dim)
        self.difficulty_layer = nn.Linear(3 * latent_dim, 1)
        self.positive_evidence_layer = nn.Linear(latent_dim, 1)
        self.negative_evidence_layer = nn.Linear(latent_dim, 1)
        self.concept_mastery_bias = nn.Embedding(knowledge_num, 1)
        self.evidence_layer1 = nn.Linear(5 * latent_dim + 2, latent_dim)
        self.evidence_layer2 = nn.Linear(latent_dim, 1)
        self.apply(self._initialize)

        raw_scale = math.log(math.expm1(max(float(init_gap_scale), eps)))
        self.raw_gap_scale = nn.Parameter(torch.tensor(raw_scale))
        self.student_bias = nn.Embedding(student_num, 1)
        nn.init.zeros_(self.student_bias.weight)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    @property
    def device(self):
        return self.student_emb.weight.device

    def set_graphs(self, graphs):
        self.graphs = {}
        for key, value in graphs.items():
            if value.layout == torch.sparse_coo:
                self.graphs[key] = value.to(self.device).coalesce()
            else:
                self.graphs[key] = value.to(self.device)

    @staticmethod
    def _propagate(initial, graph, layers):
        outputs = [initial]
        current = initial
        for _ in range(layers):
            current = torch.sparse.mm(graph, current)
            outputs.append(current)
        return torch.stack(outputs, dim=1).mean(dim=1)

    def _channels(self):
        q_initial = torch.cat([self.item_emb.weight, self.concept_emb.weight])
        q_output = self._propagate(q_initial, self.graphs["q_graph"], self.q_layers)
        item_semantic = q_output[: self.exercise_num]
        concept = q_output[self.exercise_num :]
        response_initial = torch.cat([self.student_emb.weight, item_semantic])
        positive = self._propagate(
            response_initial, self.graphs["right_response_graph"], self.response_layers
        )
        negative = self._propagate(
            response_initial, self.graphs["wrong_response_graph"], self.response_layers
        )
        student_positive, item_positive = (
            positive[: self.student_num],
            positive[self.student_num :],
        )
        student_negative, item_negative = (
            negative[: self.student_num],
            negative[self.student_num :],
        )
        difficulty = torch.sigmoid(
            self.difficulty_layer(
                torch.cat([item_semantic, item_positive, item_negative], dim=-1)
            )
        ).squeeze(-1)
        return item_semantic, concept, student_positive, student_negative, difficulty

    def _concept_evidence(self, item_semantic, difficulty, prefix, student_id=None):
        edge_student = self.graphs[f"{prefix}_edge_student"].long()
        edge_item = self.graphs[f"{prefix}_edge_item"].long()
        edge_concept = self.graphs[f"{prefix}_edge_concept"].long()
        inverse = None
        if student_id is None:
            aggregate_size = self.student_num
            local_student = edge_student
            count = self.graphs[f"concept_{prefix}_count"]
        else:
            student_id = student_id.long()
            unique_student, inverse = torch.unique(
                student_id, sorted=True, return_inverse=True
            )
            aggregate_size = unique_student.numel()
            positions = torch.full(
                (self.student_num,), -1, dtype=torch.long, device=self.device
            )
            positions[unique_student] = torch.arange(
                aggregate_size, device=self.device
            )
            local_student = positions[edge_student]
            mask = local_student >= 0
            local_student = local_student[mask]
            edge_item = edge_item[mask]
            edge_concept = edge_concept[mask]
            count = self.graphs[f"concept_{prefix}_count"][student_id]

        flat = local_student * self.knowledge_num + edge_concept
        weight = difficulty[edge_item] if prefix == "positive" else 1.0 - difficulty[edge_item]
        numerator = torch.zeros(
            aggregate_size * self.knowledge_num,
            self.latent_dim,
            device=self.device,
        )
        denominator = torch.zeros(
            aggregate_size * self.knowledge_num, device=self.device
        )
        if flat.numel():
            numerator.index_add_(0, flat, item_semantic[edge_item] * weight.unsqueeze(-1))
            denominator.index_add_(0, flat, weight)
        evidence = numerator.reshape(
            aggregate_size, self.knowledge_num, self.latent_dim
        )
        weight_sum = denominator.reshape(aggregate_size, self.knowledge_num)
        evidence = evidence / (weight_sum.unsqueeze(-1) + self.eps)
        if inverse is not None:
            evidence = evidence[inverse]
            weight_sum = weight_sum[inverse]
        return evidence, count.float(), weight_sum

    def _mastery(self, student_id, channels):
        item_semantic, concept, student_positive, student_negative, difficulty = channels
        student_id = student_id.long()
        positive_local, positive_count, _ = self._concept_evidence(
            item_semantic, difficulty, "positive", student_id
        )
        negative_local, negative_count, _ = self._concept_evidence(
            item_semantic, difficulty, "negative", student_id
        )
        h_pos = student_positive[student_id].unsqueeze(1)
        h_neg = student_negative[student_id].unsqueeze(1)
        v = concept.unsqueeze(0).expand(len(student_id), -1, -1)
        global_positive = F.softplus(self.positive_evidence_layer(h_pos * v)).squeeze(-1)
        global_negative = F.softplus(self.negative_evidence_layer(h_neg * v)).squeeze(-1)
        global_logit = (
            global_positive
            - global_negative
            + self.concept_mastery_bias.weight.reshape(1, -1)
        )
        evidence_input = torch.cat(
            [
                positive_local,
                negative_local,
                v,
                positive_local * v,
                negative_local * v,
                torch.log1p(positive_count).unsqueeze(-1),
                torch.log1p(negative_count).unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = F.relu(self.evidence_layer1(evidence_input))
        local_logit = self.evidence_layer2(hidden).squeeze(-1)
        return torch.sigmoid(global_logit + local_logit)

    def decode(self, mastery, difficulty, q_mask, student_id):
        q_mask = q_mask.to(dtype=mastery.dtype, device=mastery.device)
        average = (mastery * q_mask).sum(dim=1) / (q_mask.sum(dim=1) + self.eps)
        logit = F.softplus(self.raw_gap_scale) * (average - difficulty)
        logit = logit + self.student_bias(student_id.long()).squeeze(-1)
        return torch.sigmoid(logit)

    def forward(self, student_id, exercise_id, q_mask):
        channels = self._channels()
        mastery = self._mastery(student_id, channels)
        difficulty = channels[-1][exercise_id.long()]
        return self.decode(mastery, difficulty, q_mask, student_id)

    @torch.no_grad()
    def mastery_matrix(self, batch_size=512):
        self.eval()
        channels = self._channels()
        chunks = []
        for start in range(0, self.student_num, batch_size):
            ids = torch.arange(
                start, min(start + batch_size, self.student_num), device=self.device
            )
            chunks.append(self._mastery(ids, channels))
        return torch.cat(chunks).cpu().numpy()

