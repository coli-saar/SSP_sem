"""
model for script parsing as a sequence labeling task
"""
from typing import Dict, List
import os

import numpy as np
from scipy.stats import pearsonr
import dill
import torch
from transformers import XLNetModel
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from crf import ConditionalRandomField
# from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules.attention import DotProductAttention
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.predictors import Predictor

from arks_allennlp_utils import AverageF1, to_categorical
from arks_allennlp_utils import supply_token_indices
from misc import EasyPlot
from global_constants import CONST
from data import InScriptSequenceLabelingReader


class SequenceLabelingScriptParser(Model):
    def __init__(self, hypers, vocab, configurations, preceeds, event_indices=None, participant_indices=None):
        """
        :param hypers:
                - lr;
                - dropout;
                - weight decay
                - dimensions
        """
        super(SequenceLabelingScriptParser, self).__init__(vocab)
        self.configurations = configurations
        self.preceeds = preceeds
        self.n_labels = vocab.get_vocab_size('scr_labels')
        self.tagger_type = configurations.tagger_type
        self.first_feature_size = self.configurations.encoder_hidden_size

        self.treebank_embedding_mode = False
        if len(self.configurations.corpora) > 1:
            ''' activate treebank embedding '''
            self.treebank_embedding_mode = True
            self.n_treebanks = len(self.configurations.corpora)
            size = [self.n_treebanks, hypers.corpus_embedding_dim]
            self.treebank_embeddings = torch.nn.Parameter(
                data=torch.randn(size=size, device=self.configurations.device) * .1,
                requires_grad=True)
            self.first_feature_size += hypers.corpus_embedding_dim

        ''' model components '''
        self.sequence_encoder = XLNetModel.from_pretrained('xlnet-base-cased')
        if configurations.freeze:
            self.sequence_encoder.requires_grad_(False)
        self.drop_out = torch.nn.Dropout(hypers.dropout)

        if self.tagger_type == 'none':
            self.linear_classifier = torch.nn.Linear(in_features=self.first_feature_size,
                                                     out_features=self.n_labels)
        elif self.tagger_type == 'lstm':
            self.tagger = torch.nn.LSTM(input_size=self.first_feature_size, batch_first=True,
                                        hidden_size=hypers.tagger_dim, num_layers=1, bidirectional=True)
            self.linear_classifier = torch.nn.Linear(in_features=hypers.tagger_dim * 2,
                                                     out_features=self.n_labels)
        elif self.tagger_type == 'crf':
            self.tagger = ConditionalRandomField(num_tags=self.n_labels, constraints=self._allowed_transitions())
            self.linear_feature_extractor = torch.nn.Linear(in_features=self.first_feature_size,
                                                            out_features=self.n_labels)

        elif self.tagger_type == 'lstm-crf':
            self.lstm = torch.nn.LSTM(input_size=self.first_feature_size, batch_first=True,
                                      hidden_size=hypers.tagger_dim, num_layers=1, bidirectional=True)
            self.linear_feature_extractor = torch.nn.Linear(in_features=hypers.tagger_dim * 2,
                                                            out_features=self.n_labels)
            self.tagger = ConditionalRandomField(num_tags=self.n_labels, constraints=self._allowed_transitions())

        elif self.tagger_type == 'att-lstm':
            self._attention = DotProductAttention()
            self.tagger = torch.nn.LSTM(input_size=self.first_feature_size, batch_first=True,
                                        hidden_size=hypers.tagger_dim, num_layers=1, bidirectional=True)
            self.linear_classifier = torch.nn.Linear(in_features=hypers.tagger_dim * 2,
                                                     out_features=self.n_labels)

        if 'crf' in self.tagger_type:
            self.crf_transitions = None
        ''' metrics'''
        self.accuracy = CategoricalAccuracy()
        self.avg_F1 = AverageF1(self.n_labels)
        self.valid_F1s = [self.avg_F1]
        if 'events' in self.configurations.clustering_mode:
            self.event_F1 = AverageF1(self.n_labels, valid_classes=event_indices)
            self.valid_F1s.append(self.event_F1)
        if 'participants' in self.configurations.clustering_mode:
            self.participant_F1 = AverageF1(self.n_labels, valid_classes=participant_indices)
            self.valid_F1s.append(self.participant_F1)

    @staticmethod
    def _advanced_feature_select(features, valid_indices, max_seq_len):
        dense_feature_shape = features.size()[0], max_seq_len, features.size()[-1]
        batch_size, _, feature_dim = dense_feature_shape
        dense_source_mask = features.new_zeros([batch_size, max_seq_len])
        feature_selected = features.new_zeros(size=dense_feature_shape, dtype=torch.float)
        for i in range(batch_size):
            feature_selected[i, :len(valid_indices[i]), :] = features[i].\
                index_select(dim=0, index=feature_selected.new_tensor(valid_indices[i], dtype=torch.long))
            dense_source_mask[i, :len(valid_indices[i])] += 1
        dense_source_mask.bool()
        return feature_selected, dense_source_mask

    def export_crf_transitions(self, output_folder, thres=2.15, edge=' -> '):
        """ export the crf transition probabilities as graphviz code """
        assert 'crf' in self.tagger_type
        by_scenario_labels = dict()

        def _node_code(model, _scenario, _index):
            inst_count = model.avg_F1.instance_counts[_index]
            thickness = np.log(inst_count) if inst_count > 0 else 0.001
            name = by_scenario_labels[_scenario][_index]
            return f' {name} [width="{thickness}"] '

        for index in range(self.vocab.get_vocab_size('scr_labels')):
            full_label = self.vocab.get_token_from_index(index, 'scr_labels')
            scenario = InScriptSequenceLabelingReader.scenario_of_label(full_label)
            label = full_label[:-(len(scenario) + 1)]
            if '_' in label:
                label = label[label.find('_') + 1:]
            else:
                label = label[1:]
            label = label.replace('/', '_')
            if scenario not in by_scenario_labels:
                by_scenario_labels[scenario] = dict()
            by_scenario_labels[scenario][index] = label
        for scenario in by_scenario_labels:
            nodes = ';'.join(
                [_node_code(self, scenario, index) for index in list(by_scenario_labels[scenario].keys())
                 if '@' not in scenario]
                + ['START', 'END'])
            edges = list()
            for label_i in by_scenario_labels[scenario]:
                for label_j in by_scenario_labels[scenario]:
                    weight = 2. * np.exp(self.tagger.transitions[label_i, label_j].detach().cpu().numpy())
                    if weight > thres:
                        edges.append(f'{by_scenario_labels[scenario][label_i]} '
                                     f'{edge} {by_scenario_labels[scenario][label_j]}'
                                     f' [penwidth={weight}]')
            # start and end edges
            for label_i in by_scenario_labels[scenario]:
                weight = 2. * np.exp(self.tagger.start_transitions[label_i].detach().cpu().numpy())
                if weight > thres:
                    edges.append(f'START {edge} {by_scenario_labels[scenario][label_i]}'
                                 f' [penwidth={weight}]')
                weight = 2. * np.exp(self.tagger.end_transitions[label_i].detach().cpu().numpy())
                if weight > thres:
                    edges.append(f'{by_scenario_labels[scenario][label_i]} {edge} END'
                                 f' [penwidth={weight}]')
            edge_string = ';\n'.join(edges + [''])
            with open(f'{output_folder}_{scenario}.gv', 'w') as viz_out:
                viz_out.write(f'digraph {scenario} \n')
                viz_out.write('{\n')
                viz_out.write('	node [fixedsize=true regular=true shape=circle];  ')
                viz_out.write('rank=same;' + nodes + ' \n')
                viz_out.write(edge_string)
                viz_out.write('}')

    def forward(self, story, scenario: Dict[str, Dict[str, torch.Tensor]],
                label_indices, squeezed_labels=None, scr_labels=None) -> Dict[str, torch.Tensor]:
        """"""
        '-- acquire representations --'
        source_mask = util.get_text_field_mask(story)
        loss, predicted_labels = 0., 0.
        scr_labels_tensor = scr_labels['scr_labels']['tokens']

        # shape: batch, len, dim
        encoded_sequence = self.sequence_encoder(story['words']['token_ids'])[0]

        if self.treebank_embedding_mode:
            def _get_treebank_indices():
                batch_size = scenario['scenarios']['tokens'].size()[0]
                scenario_names = \
                    [self.vocab.get_token_from_index(
                        namespace='scenarios',
                        index=int(scenario['scenarios']['tokens'][ind]))
                     for ind in range(batch_size)]
                corpora_names = [scenario_name[scenario_name.index('_') + 1:] for scenario_name in scenario_names]
                treebank_indices_list = [self.configurations.corpora.index(c) for c in corpora_names]
                return torch.tensor(data=treebank_indices_list).to(self.configurations.device)
            # shape: b
            treebank_indices = _get_treebank_indices()
            # shape: b * corpus_embedding_dim
            batch_treebank_embeddings_pre = self.treebank_embeddings.index_select(dim=0, index=treebank_indices)
            seq_len = encoded_sequence.size()[1]
            batch_treebank_embeddings = batch_treebank_embeddings_pre.unsqueeze(1).repeat(1, seq_len, 1)
            encoded_sequence = torch.cat([encoded_sequence, batch_treebank_embeddings], dim=-1)

        # shape: batch, len, 2*dim
        encoded_sequence_dropped = self.drop_out(encoded_sequence)

        if self.tagger_type == 'none':
            logits = self.linear_classifier(encoded_sequence_dropped)
            predicted_labels = logits.argmax(dim=-1)
            if scr_labels is not None:
                # filter out none labels
                valid_label_mask = scr_labels_tensor != self.vocab.get_token_index(CONST.null_label, 'scr_labels')
                final_mask = source_mask * valid_label_mask
                self.accuracy(predictions=logits, gold_labels=scr_labels_tensor, mask=final_mask)
                for avg_F1 in self.valid_F1s:
                    avg_F1(predictions=logits, gold_labels=scr_labels_tensor, mask=final_mask)
                loss = util.sequence_cross_entropy_with_logits(logits, scr_labels_tensor, final_mask)

        elif self.tagger_type in ['lstm', 'att-lstm']:
            max_tag_seq_len = squeezed_labels['scr_labels']['tokens'].size()[-1]
            # prepare squeezed data for tagger
            if self.configurations.select_index:
                features_selected, final_mask = SequenceLabelingScriptParser.\
                    _advanced_feature_select(encoded_sequence, label_indices, max_tag_seq_len)
            else:
                features_selected, final_mask = encoded_sequence_dropped, source_mask

            classification_features = self.tagger(features_selected)[0]
            logits = self.linear_classifier(classification_features)
            predicted_labels = logits.argmax(dim=-1)
            if scr_labels is not None:
                if self.configurations.select_index:
                    squeezed_labels_tensor = squeezed_labels['scr_labels']['tokens']
                    self.accuracy(predictions=logits, gold_labels=squeezed_labels_tensor, mask=final_mask)
                    for avg_F1 in self.valid_F1s:
                        avg_F1(predictions=logits, gold_labels=squeezed_labels_tensor, mask=final_mask.to(torch.bool))
                    loss = util.sequence_cross_entropy_with_logits(
                        logits, squeezed_labels_tensor, final_mask.to(torch.bool))
                else:
                    scr_labels_tensor = scr_labels['scr_labels']['tokens']
                    self.accuracy(predictions=logits, gold_labels=scr_labels_tensor, mask=final_mask)
                    for avg_F1 in self.valid_F1s:
                        avg_F1(predictions=logits, gold_labels=scr_labels_tensor, mask=final_mask.to(torch.bool))
                    loss = util.sequence_cross_entropy_with_logits(
                        logits, scr_labels['scr_labels']['tokens'], final_mask.to(torch.bool))

        elif self.tagger_type in ['crf', 'lstm-crf']:
            max_tag_seq_len = squeezed_labels['scr_labels']['tokens'].size()[-1]
            if self.tagger_type == 'lstm-crf':
                # size: batch, denselen, n_labels
                by_label_features = self.linear_feature_extractor(self.lstm(encoded_sequence_dropped)[0])
            else:  # self.tagger_type == 'crf':
                # size: batch, denselen, n_labels
                by_label_features = self.linear_feature_extractor(encoded_sequence_dropped)
            features_select, final_mask = SequenceLabelingScriptParser.\
                _advanced_feature_select(by_label_features, label_indices, max_tag_seq_len)

            # shape: batch * denselen.
            predictions = self.tagger.viterbi_tags(logits=features_select, mask=final_mask)
            predicted_labels = features_select.new_zeros(size=final_mask.size(), dtype=torch.int64)
            for i, (labels, likelihood) in enumerate(predictions):
                predicted_labels[i][:len(labels)] = torch.tensor(labels, dtype=torch.int64)
            # shape: batch * denselen * n_labels.
            pseudo_logits = to_categorical(predicted_labels, self.n_labels)
            if scr_labels is not None:
                squeezed_labels_tensor = squeezed_labels['scr_labels']['tokens']
                self.accuracy(predictions=pseudo_logits, gold_labels=squeezed_labels_tensor, mask=final_mask)
                for avg_F1 in self.valid_F1s:
                    avg_F1(predictions=pseudo_logits,
                           gold_labels=squeezed_labels_tensor, mask=final_mask.to(torch.bool))
                log_likelihood = self.tagger(
                    inputs=features_select, tags=squeezed_labels_tensor, mask=final_mask.to(torch.bool))
                avg_log_likelihood = log_likelihood / torch.sum(source_mask)
                loss = -1.0 * avg_log_likelihood

        output = dict()
        output['predictions'] = predicted_labels
        if scr_labels is not None:
            output['loss'] = loss

        return output

    @classmethod
    def from_checkpoint(cls, check_point_config, configurations, preceeds):
        combination_s = dill.load(open(check_point_config['combination_file'], 'rb'))
        combination = combination_s[check_point_config['index']]
        vocabulary = Vocabulary.from_files(check_point_config['vocab_folder'])
        event_labels = [i for i in range(vocabulary.get_vocab_size('scr_labels'))
                        if '#' in vocabulary.get_token_from_index(i, 'scr_labels')]
        participant_labels = [i for i in range(vocabulary.get_vocab_size('scr_labels'))
                              if '@' in vocabulary.get_token_from_index(i, 'scr_labels')]
        model = cls(
            hypers=combination, vocab=vocabulary, configurations=configurations,
            participant_indices=participant_labels, event_indices=event_labels, preceeds=preceeds)
        model.load_state_dict(torch.load(open(check_point_config['model_path'], 'rb')))
        return model

    def get_crf_transitions_for_heatmap(self):
        """ export the crf transition potentials for heatmap drawing """
        assert 'crf' in self.tagger_type
        by_scenario_labels = dict()
        crf_blueprints = dict()
        for index in range(self.vocab.get_vocab_size('scr_labels')):
            full_label = self.vocab.get_token_from_index(index, 'scr_labels')
            scenario = InScriptSequenceLabelingReader.scenario_of_label(full_label)
            label = full_label[:-(len(scenario) + 1)]
            if '_' in label:
                label = label[label.find('_') + 1:]
            else:
                label = label[1:]
            label = label.replace('/', '_')
            if scenario not in by_scenario_labels:
                by_scenario_labels[scenario] = dict()
            by_scenario_labels[scenario][index] = label
        for scenario in by_scenario_labels:
            if '@' in scenario:
                continue
            names = [by_scenario_labels[scenario][index] for index in sorted(list(by_scenario_labels[scenario].keys()))]
            x_names = names + ['START']
            y_names = names + ['END']
            n_labels = len(by_scenario_labels[scenario].keys())
            transition_potentials = np.zeros(shape=[n_labels + 1, n_labels + 1])
            for i in range(n_labels):
                for j in range(n_labels):
                    label_i = sorted(list(by_scenario_labels[scenario].keys()))[i]
                    label_j = sorted(list(by_scenario_labels[scenario].keys()))[j]
                    transition_potentials[i, j] = \
                        np.exp(self.tagger.transitions[label_i, label_j].detach().cpu().numpy())
                transition_potentials[-1, i] = self.tagger.start_transitions[i]
                transition_potentials[i, -1] = self.tagger.end_transitions[i]
            crf_blueprints[scenario] = (transition_potentials, x_names, y_names,
                                        os.path.join(*['.', 'heatmaps', scenario]), scenario)
        return crf_blueprints
    
    def get_detailed_metric(self):
        """
        extracts detail metrics from the model. so call only when this makes sense, i.e. the model has processed
        instances and the metrics have not been reset.
        :return:
        """
        detailed_metrics = dict()
        for class_index in self.avg_F1.valid_classes:
            class_name = self.vocab.get_token_from_index(index=class_index, namespace='scr_labels')
            scenario = InScriptSequenceLabelingReader.scenario_of_label(class_name)
            if scenario not in detailed_metrics:
                detailed_metrics[scenario] = dict()
            if 'macro_F1' not in detailed_metrics[scenario]:
                scenario_label_indices = \
                    [i for i in self.avg_F1.valid_classes if InScriptSequenceLabelingReader.
                        scenario_of_label(self.vocab.get_token_from_index(index=i, namespace='scr_labels')) == scenario]
                macro_F1, micro_F1 = self.avg_F1.get_customize_metric(valid_classes=scenario_label_indices)
                detailed_metrics[scenario]['macro_F1'] = macro_F1
                detailed_metrics[scenario]['micro_F1'] = micro_F1
            p, r, f1 = self.avg_F1.by_class_F1[class_index].get_metric()
            count = self.avg_F1.instance_counts[class_index]
            detailed_metrics[scenario][class_name] = [p, r, f1, count]
        return detailed_metrics

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        macro_F1, micro_F1 = self.avg_F1.get_metric(reset)
        metrics = {'accuracy': self.accuracy.get_metric(reset),
                   'macro_F1': macro_F1,
                   'micro_F1': micro_F1}
        if 'events' in self.configurations.clustering_mode:
            E_macro_F1, E_micro_F1 = self.event_F1.get_metric(reset)
            metrics['events_ma_F1'], metrics['events_mi_F1'] = E_macro_F1, E_micro_F1
        if 'participants' in self.configurations.clustering_mode:
            P_macro_F1, P_micro_F1 = self.participant_F1.get_metric(reset)
            metrics['participants_ma_F1'], metrics['participants_mi_F1'] = P_macro_F1, P_micro_F1
        return metrics

    def inference(self, input_folder: str, output_file: str, dataset_reader, batch_size=8):
        """
        performs inference for all files in the input folder and exports the results to output_folder
        :return:
        """
        predictor = Predictor(self, dataset_reader)
        inf_data = dataset_reader.read(input_folder)
        pretrained_tokenizer = PretrainedTransformerTokenizer(self.configurations.pretrained_model_name)
        supply_token_indices(inf_data, 'story', pretrained_tokenizer)
        predictions = list()
        num_batches = int(np.ceil(len(inf_data) / batch_size))
        for batch_index in range(num_batches):
            predictions.extend(predictor.predict_batch_instance(
                inf_data[batch_size * batch_index: batch_size * (batch_index + 1)]))
            print('.', end='')
        # todo: output predictions
        with open(os.path.join('.', 'results_' + output_file), 'w') as inference_out:
            count_not_proceed = 0
            for index, prediction in enumerate(predictions):
                valid_label_pointer = 0
                for token_index in range(len(inf_data[index].fields['story'].tokens)):
                    token = inf_data[index].fields['story'].tokens[token_index].text
                    reference_label = inf_data[index].fields['scr_labels'].tokens[token_index].text
                    if token_index in inf_data[index].fields['label_indices'].metadata:
                        predicted_label = self.vocab.get_token_from_index(
                            index=prediction['predictions'][valid_label_pointer], namespace='scr_labels')
                        valid_label_pointer += 1
                    else:
                        predicted_label = 'n/a'
                    correct = str(predicted_label == reference_label) if predicted_label != 'n/a' else 'n/a'
                    line = ' '.join([token, reference_label, predicted_label, correct, '\n'])
                    inference_out.write(line)
                for event_index in range(len(prediction['predictions']) - 1):
                    [event_1, event_2] = \
                        [self.vocab.get_token_from_index(prediction['predictions'][_eind], 'scr_labels')
                         for _eind in [event_index, event_index + 1]]
                    scenario = inf_data[index].fields['scenario'].tokens[0].text
                    if '_' in scenario:
                        scenario = scenario[:scenario.index('_')]
                    # if (event_1, event_2) not in self.preceeds[scenario] and 'none' not in [event_1, event_2]:
                    #     count_not_proceed += 1
                    #     print(f'{count_not_proceed}   {event_1}, {event_2}')
        metrics = self.get_detailed_metric()
        f1, n = list(), list()
        with open(os.path.join('.', 'metrics_' + output_file), 'w') as metric_out:
            for scenario, scenario_metric in metrics.items():
                if 'UNK' in scenario:
                    continue
                for key, metric in scenario_metric.items():
                    line = ','.join([scenario, key, str(metric)])
                    metric_out.write(line + '\n')
                    if '#' in key and 'irregular' not in key:
                        f1.append(metric[2])
                        n.append(metric[3])
        print('r, p = ' + str(pearsonr(f1, n)))
        EasyPlot.plot_line_graph(x=n, y=f1, x_name='#instance in cluster', y_name='F1', out_name='pearson_original')

        # self.export_crf_transitions(output_folder=os.path.join('.', ''))
        # transitions = self.get_crf_transitions_for_heatmap()
        # self.crf_transitions = transitions
        # for scenario, s in self.crf_transitions.items():
        #     EasyPlot.plot_heatmap(s[0], s[1], s[2], s[3], s[4])

        return inf_data

    def _allowed_transitions(self):
        """
        return the transitions allowed for a linear crf. start and end transitions are att
        :return:
        """
        allowed_transitions = list()
        num_labels = self.vocab.get_vocab_size('scr_labels')
        for i in range(num_labels):
            for j in range(num_labels):
                if self._is_allowed_transition(i, j):
                    allowed_transitions.append((i, j))
        for i in range(num_labels):
            # transitions from START
            allowed_transitions.append((num_labels, i))
            # transitions to END
            allowed_transitions.append((i, num_labels + 1))
        return allowed_transitions

    def _is_allowed_transition(self, index1, index2):
        """ evaluate whether a transition between two states is allowed in the crf. basically only transitions between
         states from the same scenario are allowed """
        label1 = self.vocab.get_token_from_index(index1, 'scr_labels')
        label2 = self.vocab.get_token_from_index(index2, 'scr_labels')
        return InScriptSequenceLabelingReader.scenario_of_label(label1) \
            == InScriptSequenceLabelingReader.scenario_of_label(label2)

    def test(self, test_data, dataset_reader, batch_size=8):
        """
        acquire test metrics
        :param test_data:
        :param dataset_reader:
        :param batch_size:
        :return:
        """
        # reset metrics is not necessary as allenNLP resets metrics in trainer.train()
        predictor = Predictor(self, dataset_reader)
        pretrained_tokenizer = PretrainedTransformerTokenizer(self.configurations.pretrained_model_name)
        supply_token_indices(test_data, 'story', pretrained_tokenizer)
        predictions = list()
        num_batches = int(np.ceil(len(test_data) / batch_size))
        for batch_index in range(num_batches):
            predictions.extend(predictor.predict_batch_instance(
                test_data[batch_size * batch_index: batch_size * (batch_index + 1)]))
            print('.', end='')
        metrics = self.get_metrics(reset=True)
        test_metrics = {'test_' + key: metrics[key] for key in metrics}
        # todo: write a loss tracker for testing
        test_metrics['test_loss'] = 0.
        return test_metrics
