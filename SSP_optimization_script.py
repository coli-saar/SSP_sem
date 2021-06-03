"""
configurations different in each execution.
No, I'm NOT a fan of config files.
"""
import enum
import os

from optimization import ScriptRepresentationLearningMetaOptimizer
from model import SequenceLabelingScriptParser
from data import InScriptSequenceLabelingReader


class ExecutionSettings:

    class ExecutionMode(enum.Enum):
        OPTIMIZATION = 'optimization'
        TEST = 'test'

    execution_mode = ExecutionMode.OPTIMIZATION

    select_index = True

    ''' one of {'none', 'lstm', 'crf', 'lstm-crf'}, specifies the tagging layer.'''
    tagger_type = 'lstm'

    ''' subset of {inscript, descript, backtranslation}. the list should match the data '''
    corpora = ['inscript', 'descript']

    ''' a non-empty subset of {'events', 'participants'}, indicating which instances will be loaded. '''''
    clustering_mode = ['events', 'participants']

    ''' one of 'normal', 'regular_only', 'regular_identification', 
    check the doc of InScriptSequenceLabeling Reader '''
    loading_mode = 'normal'

    tag = ''.join([s[0] for s in clustering_mode]) + '_' + tagger_type

    train_data_dir = os.path.join('.', 'data_train') if execution_mode == ExecutionMode.OPTIMIZATION \
        else os.path.join('.', 'toy')
    val_data_dir = os.path.join('.', 'data_val') if execution_mode == ExecutionMode.OPTIMIZATION \
        else os.path.join('.', 'toy')
    test_data_dir = os.path.join('.', 'data_test') if execution_mode == ExecutionMode.OPTIMIZATION \
        else os.path.join('.', 'toy')

    pretrained_model_name = 'xlnet-base-cased'
    encoder_hidden_size = 768

    freeze = True
    device = 0
    num_trials = 10 if execution_mode == ExecutionMode.OPTIMIZATION else 2
    patience = 10 if execution_mode == ExecutionMode.OPTIMIZATION else 3
    max_epochs = 80 if execution_mode == ExecutionMode.OPTIMIZATION else 1

    param_domains = {'batch_size': [4, 15] if freeze is False else 32,
                     'lr': 1.2e-4 if freeze is True else [5e-6, 1e-4],
                     'l2': 1.52e-4,
                     'clip': 1.84,
                     'dropout': 0.167 if execution_mode == ExecutionMode.OPTIMIZATION else [0., 0.],
                     'tagger_dim': [256, 1023],
                     'corpus_embedding_dim': [256, 2047]}

    if execution_mode == ExecutionMode.TEST:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == '__main__':
    meta_optimizer = ScriptRepresentationLearningMetaOptimizer(configurations=ExecutionSettings,
                                                               model=SequenceLabelingScriptParser,
                                                               dataset_reader=InScriptSequenceLabelingReader)
    meta_optimizer.search(test_mode=True)  # (ExecutionSettings.execution_mode == ExecutionSettings.ExecutionMode.TEST))
