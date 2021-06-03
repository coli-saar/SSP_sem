"""
utils to facilitate AllenNLP based projects
this file collects stuff written for allenNLP >= v1.0
"""
from typing import Optional, Union, Tuple, Dict, List

import torch
from allennlp.training.metrics import Metric, F1Measure


def to_categorical(y: torch.Tensor, num_classes) -> torch.Tensor:
    if y.dim() == 1:
        return torch.eye(num_classes, dtype=torch.long, device=y.device)[y]
    elif y.dim() == 2:
        b, le = y.size()
        r = y.new_zeros(size=[b, le, num_classes])
        for i in range(b):
            r[i] = to_categorical(y[i], num_classes)
        return r
    else:
        raise IndexError('This function only processes input with dimensions 1 or 2.')


def supply_token_indices(instances, text_field_name: str, pretrained_tokenizer):
    """
    attach text_id s to text_field tokens to patch the behavior of allenNLP's pretrained transformer token indexers
    :param instances:
    :param text_field_name:
    :param pretrained_tokenizer:
    :return:
    """
    for instance in instances:
        for token in instance.fields[text_field_name]:
            token.text_id = pretrained_tokenizer.tokenizer.convert_tokens_to_ids(token.text)


class AverageF1(Metric):
    """
    track F1 by classes to allow evaluation of micro (by instance averaged) and macro (by class averaged) F1.
    """
    def __init__(self, n_clusters, valid_classes: List = None):
        """
        :param n_clusters:
        :param valid_classes: the classes whose F1 should be tracked. If unspecified, all classes are tracked.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.instance_counts = dict()
        self.by_class_F1 = dict()
        self.valid_classes = valid_classes or list(range(self.n_clusters))
        for class_label in self.valid_classes:
            self.by_class_F1[class_label] = F1Measure(positive_label=class_label)

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]):
        predictions, gold_labels, mask = predictions.detach(), gold_labels.detach(), mask.detach()
        for class_label in self.valid_classes:
            count = int(torch.sum((gold_labels == class_label) * mask, dim=list(range(gold_labels.dim())))
                        .detach().cpu())
            if class_label not in self.instance_counts:
                self.instance_counts[class_label] = 0
            self.instance_counts[class_label] += count
            self.by_class_F1[class_label](predictions, gold_labels, mask)

    def reset(self) -> None:
        self.instance_counts = dict()
        self.by_class_F1 = dict()
        for class_label in self.valid_classes:
            self.by_class_F1[class_label] = F1Measure(positive_label=class_label)

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        macro_F1, micro_F1 = self.get_customize_metric(valid_classes=self.valid_classes)
        if reset:
            self.reset()
        return macro_F1, micro_F1

    def get_customize_metric(self, valid_classes: List):
        cumulated_micro_F1 = 0.
        cumulated_macro_F1 = 0.
        n_valid_classes = 0.
        for class_label in valid_classes:
            if self.instance_counts[class_label] > 0:
                p, r, f1 = self.by_class_F1[class_label].get_metric(reset=False)
                cumulated_micro_F1 += f1 * self.instance_counts[class_label]
                cumulated_macro_F1 += f1
                n_valid_classes += 1
        n_valid_instances = sum([self.instance_counts[class_label] for class_label in valid_classes])
        micro_F1 = \
            cumulated_micro_F1 / n_valid_instances if n_valid_instances > 0. else 0.0
        macro_F1 = cumulated_macro_F1 / n_valid_classes if n_valid_classes > 0. else 0.0
        return macro_F1, micro_F1
