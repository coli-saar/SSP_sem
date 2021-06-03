"""
perform inference and analysis of the results
"""
import enum
import os


from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from data import InScriptSequenceLabelingReader
from model import SequenceLabelingScriptParser


execution_mode = 'optimization'
inf_data_dir = os.path.join(*['..', 'exe', '_1_SSP_DA_lstm_R1', 'data_val']) if execution_mode == 'optimization' \
    else os.path.join('.', 'toy')

''' =============== checkpoint details ================= '''
# note: Remember to copy the execution settings
'''----------------------------------------'''
test_folder = os.path.join(*['..', 'script_parsing_checkpoints', '2_lstm_ep_freeze_1'])
check_point_test = {
    'combination_file': os.path.join(test_folder, 'hyper_combs_ep_lstm'),
    'index': 1,
    'vocab_folder': os.path.join(test_folder, 'vocab_opt_ep_lstm_1'),
    'model_path': os.path.join(test_folder, 'best.th'),
}
'''----------------------------------------'''
none_freeze_e_folder = os.path.join(*['..', 'script_parsing_checkpoints', '0_none_e_freeze_11'])
check_point_none_e_f = {
    'combination_file': os.path.join(none_freeze_e_folder, 'hyper_combs_opt_[\'events\']'),
    'index': 11,
    'vocab_folder': os.path.join(none_freeze_e_folder, 'vocab_opt_[\'events\']_11'),
    'model_path': os.path.join(none_freeze_e_folder, 'best.th'),
}
'''----------------------------------------'''
lstm_freeze_e_folder = os.path.join(*['..', 'script_parsing_checkpoints', '1_lstm_freeze_9'])
check_point_lstm_e_f = {
    'combination_file': os.path.join(lstm_freeze_e_folder, 'hyper_combs_opt_[\'events\']'),
    'index': 9,
    'vocab_folder': os.path.join(lstm_freeze_e_folder, 'vocab_opt_[\'events\']_9'),
    'model_path': os.path.join(lstm_freeze_e_folder, 'best.th'),
}
'''----------------------------------------'''
crf_freeze_e_folder = os.path.join(*['..', 'script_parsing_checkpoints', '6_crf_freeze_10'])
check_point_crf_e_f = {
    'combination_file': os.path.join(crf_freeze_e_folder, 'hyper_combs_opt_[\'events\']'),
    'index': 10,
    'vocab_folder': os.path.join(crf_freeze_e_folder, 'vocab_opt_[\'events\']_10'),
    'model_path': os.path.join(crf_freeze_e_folder, 'best.th'),
}
'''----------------------------------------'''
lstm_crf_freeze_e_folder = os.path.join(*['..', 'script_parsing_checkpoints', '2_lstm-crfpp_18'])
check_point_lstm_crf_e_f = {
    'combination_file': os.path.join(lstm_crf_freeze_e_folder, 'hyper_combs_e_lstm-crf'),
    'index': 18,
    'vocab_folder': os.path.join(lstm_crf_freeze_e_folder, 'vocab_e_lstm-crf_18'),
    'model_path': os.path.join(lstm_crf_freeze_e_folder, 'best.th'),
}
'''----------------------------------------'''
lstm_crfpp_freeze_e_folder = os.path.join(*['..', 'script_parsing_checkpoints', '7_lstmcrfpp_freeze_10'])
check_point_lstm_crfpp_e_f = {
    'combination_file': os.path.join(lstm_crfpp_freeze_e_folder, 'hyper_combs_e_lstm-crf'),
    'index': 10,
    'vocab_folder': os.path.join(lstm_crfpp_freeze_e_folder, 'vocab_e_lstm-crf_10'),
    'model_path': os.path.join(lstm_crfpp_freeze_e_folder, 'best.th'),
}
lstm_ep_freeze_e_folder = os.path.join(*['..', 'script_parsing_checkpoints', '7_lstm_ep_f_8'])
check_point_lstm_ep_f = {
    'combination_file': os.path.join(lstm_ep_freeze_e_folder, 'hyper_combs_pe_lstm'),
    'index': 8,
    'vocab_folder': os.path.join(lstm_ep_freeze_e_folder, 'vocab_pe_lstm_8'),
    'model_path': os.path.join(lstm_ep_freeze_e_folder, 'best.th'),
}

lstm_treebank_acl_folder = os.path.join(*['..', 'sspacl2021checkpoint'])
lstm_treebank_acl = {
    'combination_file': os.path.join(lstm_treebank_acl_folder, 'hyper_combs_ep_lstm'),
    'index': 15,
    'vocab_folder': os.path.join(lstm_treebank_acl_folder, 'vocab_ep_lstm_15'),
    'model_path': os.path.join(lstm_treebank_acl_folder, 'best.th'),
}

lstm_acl_folder = os.path.join(*['..', 'secondssp2021aclcheckpointlstm'])
lstm_acl = {
    'combination_file': os.path.join(lstm_acl_folder, 'hyper_combs_ep_lstm'),
    'index': 5,
    'vocab_folder': os.path.join(lstm_acl_folder, 'vocab_ep_lstm_5'),
    'model_path': os.path.join(lstm_acl_folder, 'best.th'),
}


''' note: use the correct settings '''


class ExecutionSettings:

    class ExecutionMode(enum.Enum):
        OPTIMIZATION = 'optimization'
        TEST = 'test'

    execution_mode = ExecutionMode.OPTIMIZATION

    ''' one of {'none', 'lstm', 'crf', 'lstm-crf'}, specifies the tagging layer.'''
    tagger_type = 'lstm'

    ''' subset of {inscript, descript, backtranslation}. the list should match the data '''
    corpora = ['inscript', 'descript']

    ''' a non-empty subset of {'events', 'participants'}, indicating which instances will be loaded. '''''
    clustering_mode = ['events', 'participants']

    ''' one of 'normal', 'regular_only', 'regular_identification', check the doc of InScriptSequenceLabeling Reader '''
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
    device = 1
    num_trials = 20 if execution_mode == ExecutionMode.OPTIMIZATION else 2
    patience = 10 if execution_mode == ExecutionMode.OPTIMIZATION else 3
    max_epochs = 200 if execution_mode == ExecutionMode.OPTIMIZATION else 1

    param_domains = {'batch_size': [4, 15] if freeze is False else [16, 127],
                     'lr': [5e-5, 5e-4] if freeze is True else [5e-6, 1e-4],
                     'l2': [1e-4, 5e-2],
                     'clip': [1, 10],
                     'dropout': [0.1, 0.9] if execution_mode == ExecutionMode.OPTIMIZATION else [0., 0.],
                     'tagger_dim': [256, 1023],
                     'corpus_embedding_dim': [128, 1023]}

    if execution_mode == ExecutionMode.TEST:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == '__main__':
    if execution_mode == 'test':
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    dataset_reader = InScriptSequenceLabelingReader(
        candidate_types=ExecutionSettings.clustering_mode,
        word_indexer={'words': PretrainedTransformerIndexer(ExecutionSettings.pretrained_model_name)})
    train_data, val_data = (dataset_reader.read(folder) for folder in
                            [ExecutionSettings.train_data_dir,
                             ExecutionSettings.val_data_dir])
    preceeds = dict()
    for instance in train_data:
        for ind in range(len(instance.fields['squeezed_labels'].tokens) - 1):
            for ind_j in range(ind):
                [event_1, event_2] = [instance.fields['squeezed_labels'].tokens[i].text for i in [ind_j, ind]]
                scenario = dataset_reader.scenario_of_label(event_1)
                if scenario not in preceeds:
                    preceeds[scenario] = dict()
                if (event_1, event_2) not in preceeds[scenario]:
                    preceeds[scenario][(event_1, event_2)] = 0
                preceeds[scenario][(event_1, event_2)] += 1
    s_preceeds = dict()
    if len(ExecutionSettings.corpora) > 1:
        for scenario in preceeds:
            s_preceeds[scenario + '_inscript'] = preceeds[scenario]
    else:
        s_preceeds = preceeds

    model = SequenceLabelingScriptParser.from_checkpoint(lstm_treebank_acl,
                                                         configurations=ExecutionSettings,
                                                         preceeds=s_preceeds).to(ExecutionSettings.device)
    model.inference(input_folder=inf_data_dir,
                    dataset_reader=dataset_reader,
                    output_file=f'{ExecutionSettings.tag}.csv')
