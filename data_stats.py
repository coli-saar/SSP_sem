"""
dataset reader for InScript as a sequence labeling task
sampler to make sure instances in each batch are from the same scenario
"""

import os

from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer

from data import InScriptSequenceLabelingReader
from global_constants import CONST


mode = ['events']
pretrained_model_name = 'xlnet-base-cased'
train_data_dir = os.path.join('.', 'data_train')
val_data_dir = os.path.join('.', 'data_val')

if __name__ == '__main__':

    dataset_reader = InScriptSequenceLabelingReader(
        candidate_types=mode,
        word_indexer={'words': PretrainedTransformerIndexer(pretrained_model_name)})
    ''' .read returns list of instances '''
    train_data, val_data = (dataset_reader.read(folder) for folder in
                            [train_data_dir, val_data_dir])
    preceeds = dict()
    for instance in train_data:
        for ind in range(len(instance.fields['squeezed_labels'].tokens) - 1):
            [event_1, event_2] = [instance.fields['squeezed_labels'].tokens[i].text for i in [ind, ind + 1]]
            scenario = InScriptSequenceLabelingReader.scenario_of_label(event_1)
            if scenario not in preceeds:
                preceeds[scenario] = dict()
            if (event_1, event_2) not in preceeds[scenario]:
                preceeds[scenario][(event_1, event_2)] = 0
            preceeds[scenario][(event_1, event_2)] += 1

    a = 1
