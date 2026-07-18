import torch
import torch.nn as nn
import torch.nn.functional as F

from model_core import ConECDCore


class ConECD(ConECDCore):
    """Concept Evidence-augmented Cognitive Diagnosis.

    The original local-evidence MLP keeps responsibility for the signed local
    correction.  A separate gate estimates only how much that correction should
    be trusted from response-independent quality features.  It is therefore not
    a hand-tuned function of interaction count.
    """

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
        local_logit_cap=1.0,
    ):
        super().__init__(
            student_num=student_num,
            exercise_num=exercise_num,
            knowledge_num=knowledge_num,
            latent_dim=latent_dim,
            q_layers=q_layers,
            response_layers=response_layers,
            eps=eps,
            init_gap_scale=init_gap_scale,
        )
        if local_logit_cap <= 0:
            raise ValueError("local_logit_cap must be positive")
        self.local_logit_cap = float(local_logit_cap)

        # Prior precision is concept- and student-specific.
        self.global_precision_layer = nn.Linear(5 * latent_dim, 1)
        # Per-response evidence quality excludes the response label, so the
        # gate cannot decide its magnitude from whether an answer is right/wrong.
        self.evidence_precision_layer1 = nn.Linear(5 * latent_dim + 2, latent_dim)
        self.evidence_precision_layer2 = nn.Linear(latent_dim, 1)
        self.global_precision_layer.apply(self._initialize)
        self.evidence_precision_layer1.apply(self._initialize)
        self.evidence_precision_layer2.apply(self._initialize)

    def _dynamic_gate(self, global_precision, local_precision):
        return local_precision / (global_precision + local_precision + self.eps)

    def _bounded_local_correction(self, local_logit):
        return self.local_logit_cap * torch.tanh(local_logit)

    def _local_precision(
        self,
        student_id,
        item_semantic,
        concept,
        student_positive,
        student_negative,
        difficulty,
        global_logit,
    ):
        """Aggregate response quality over each selected student-concept pair."""
        aggregate_size = student_id.numel()
        positions = torch.full(
            (self.student_num,), -1, dtype=torch.long, device=self.device
        )
        positions[student_id] = torch.arange(aggregate_size, device=self.device)
        precision = torch.zeros(
            aggregate_size * self.knowledge_num, device=self.device
        )

        for prefix in ("positive", "negative"):
            edge_student = self.graphs[f"{prefix}_edge_student"].long()
            edge_item = self.graphs[f"{prefix}_edge_item"].long()
            edge_concept = self.graphs[f"{prefix}_edge_concept"].long()
            local_student = positions[edge_student]
            mask = local_student >= 0
            if not mask.any():
                continue
            local_student = local_student[mask]
            edge_item = edge_item[mask]
            edge_concept = edge_concept[mask]

            item = item_semantic[edge_item]
            v = concept[edge_concept]
            h_pos = student_positive[student_id[local_student]]
            h_neg = student_negative[student_id[local_student]]
            prior_mastery = torch.sigmoid(global_logit[local_student, edge_concept])
            feature = torch.cat(
                [
                    item,
                    v,
                    item * v,
                    h_pos * v,
                    h_neg * v,
                    difficulty[edge_item].unsqueeze(-1),
                    prior_mastery.unsqueeze(-1),
                ],
                dim=-1,
            )
            hidden = F.relu(self.evidence_precision_layer1(feature))
            quality = torch.sigmoid(self.evidence_precision_layer2(hidden)).squeeze(-1)
            flat = local_student * self.knowledge_num + edge_concept
            precision.index_add_(0, flat, quality)

        return precision.reshape(aggregate_size, self.knowledge_num)

    def _mastery(self, student_id, channels):
        item_semantic, concept, student_positive, student_negative, difficulty = channels
        student_id = student_id.long()
        unique_student, inverse = torch.unique(
            student_id, sorted=True, return_inverse=True
        )
        positive_local, positive_count, _ = self._concept_evidence(
            item_semantic, difficulty, "positive", unique_student
        )
        negative_local, negative_count, _ = self._concept_evidence(
            item_semantic, difficulty, "negative", unique_student
        )
        h_pos = student_positive[unique_student].unsqueeze(1)
        h_neg = student_negative[unique_student].unsqueeze(1)
        v = concept.unsqueeze(0).expand(len(unique_student), -1, -1)
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

        global_feature = torch.cat(
            [h_pos * v, h_neg * v, v, h_pos.expand_as(v), h_neg.expand_as(v)],
            dim=-1,
        )
        global_precision = F.softplus(
            self.global_precision_layer(global_feature)
        ).squeeze(-1)
        local_precision = self._local_precision(
            unique_student,
            item_semantic,
            concept,
            student_positive,
            student_negative,
            difficulty,
            global_logit,
        )
        gate = self._dynamic_gate(global_precision, local_precision)
        local_correction = self._bounded_local_correction(local_logit)
        mastery = torch.sigmoid(global_logit + gate * local_correction)
        return mastery[inverse]

