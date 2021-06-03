import torch
import torch.nn.functional
import os
import time
import shutil

from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import ReduceOnPlateauLearningRateScheduler
from allennlp.data.dataloader import DataLoader
from allennlp.training.checkpointer import Checkpointer
from allennlp.data import Vocabulary
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer

from random_hyper_search_optimizer import RandomSearchMetaOptimizer
from misc import PrintColors
from arks_allennlp_utils import supply_token_indices


class ScriptRepresentationLearningMetaOptimizer(RandomSearchMetaOptimizer):
    """

    """
    def __init__(self, configurations, dataset_reader, model):
        domains = configurations.param_domains
        parameters = {
            'batch_size': {'domain': domains['batch_size'], 'sample_criterion': '2e', 'type': 'int'},
            'lr': {'domain': domains['lr'], 'sample_criterion': '10e', 'type': 'float'},
            'l2': {'domain': domains['l2'], 'sample_criterion': '10e', 'type': 'float'},
            'clip': {'domain': domains['clip'], 'sample_criterion': '10e', 'type': 'float'},
            'dropout': {'domain': domains['dropout'], 'sample_criterion': 'u', 'type': 'float'},
            'tagger_dim': {'domain': domains['tagger_dim'], 'sample_criterion': '2e', 'type': 'int'},
            'corpus_embedding_dim': {'domain': domains['corpus_embedding_dim'], 'sample_criterion': '2e', 'type': 'int'}
        }
        metrics = ['accuracy', 'loss', 'micro_F1', 'macro_F1']
        for mode in configurations.clustering_mode:
            metrics.extend([f'{mode}_ma_F1', f'{mode}_mi_F1'])
        ''' 
         NOTE: 'training' and 'validation' are special prefixed that allenNLP uses to log metrics in respective phases.
                DO NOT MODIFY. 
         '''
        super().__init__(parameters=parameters,
                         metric_names=[f'{phase}_{metric}' for phase in ['training', 'validation', 'test']
                                       for metric in metrics] + ['best_epoch', 'time_consumed(hrs)'],
                         num_trials=configurations.num_trials,
                         tag=configurations.tag)
        self.dataset_reader = dataset_reader
        self.configurations = configurations
        self.model = model

    def train(self, args_hpo, index):
        """
        trains the model, and return the metrics to the meta optimizer.
        :param args_hpo:
        :param index:
        :return:
        """

        PrintColors.prYellow(f'\n===== training with: {args_hpo} index={index}')
        PrintColors.prGreen(f'---- in mode: {self.configurations.execution_mode}, tag: {self.configurations.tag} ----')
        ''' ============ LOAD DATA ================================================================================ '''
        starting_time = time.time()
        dataset_reader = self.dataset_reader(
            candidate_types=self.configurations.clustering_mode,
            word_indexer={'words': PretrainedTransformerIndexer(self.configurations.pretrained_model_name)},
            mode=self.configurations.loading_mode)
        ''' .read returns list of instances '''
        train_data, val_data, test_data = (dataset_reader.read(folder) for folder in
                                           [self.configurations.train_data_dir,
                                            self.configurations.val_data_dir,
                                            self.configurations.test_data_dir])

        # count state pairs
        preceeds = dict()

        for instance in train_data:
            for ind in range(len(instance.fields['squeezed_labels'].tokens) - 1):
                [event_1, event_2] = [instance.fields['squeezed_labels'].tokens[i].text for i in [ind, ind + 1]]
                scenario = self.dataset_reader.scenario_of_label(event_1)
                if scenario not in preceeds:
                    preceeds[scenario] = dict()
                if (event_1, event_2) not in preceeds[scenario]:
                    preceeds[scenario][(event_1, event_2)] = 0
                preceeds[scenario][(event_1, event_2)] += 1

        pretrained_tokenizer = PretrainedTransformerTokenizer(self.configurations.pretrained_model_name)
        supply_token_indices(train_data + val_data, 'story', pretrained_tokenizer)

        ''' build vocabulary and associate it with datasets  '''
        vocabulary = Vocabulary.from_instances(train_data + val_data)
        train_data.index_with(vocabulary), val_data.index_with(vocabulary)

        train_data_loader = DataLoader(dataset=train_data, batch_size=args_hpo.batch_size)
        val_data_loader = DataLoader(dataset=val_data, batch_size=args_hpo.batch_size)

        ''' ============ DEFINE MODEL ============================================================================= '''
        ''' i keep .to() here instead of in model.__init__() to accomadate better abstraction '''
        event_labels = [i for i in range(vocabulary.get_vocab_size('scr_labels'))
                        if '#' in vocabulary.get_token_from_index(i, 'scr_labels')]
        participant_labels = [i for i in range(vocabulary.get_vocab_size('scr_labels'))
                              if '@' in vocabulary.get_token_from_index(i, 'scr_labels')]
        model = self.model(args_hpo, vocabulary, configurations=self.configurations,
                           preceeds=preceeds,
                           event_indices=event_labels,
                           participant_indices=participant_labels).to(self.configurations.device)

        ''' ============ DEFINE TRAINER =========================================================================== '''
        ''' -- serialization --------------------------------------------------- '''
        if not os.path.exists(os.path.join(*['.', 'models'])):
            os.mkdir(os.path.join(*['.', 'models']))
        if index == 0:
            for file in os.listdir(os.path.join(*['.', 'models'])):
                path = os.path.join(*['.', 'models', file])
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
        serialization_path = 'models_{}_{}'.format(self.configurations.tag, index)
        serialization_path_longer = os.path.join(*['.', 'models', serialization_path])
        vocab_path = 'vocab_{}_{}'.format(self.configurations.tag, index)
        vocab_dir_longer = os.path.join(*['.', 'models', vocab_path])
        if not os.path.exists(serialization_path_longer):
            os.mkdir(serialization_path_longer)
        model_checkpointer = Checkpointer(serialization_dir=serialization_path_longer, num_serialized_models_to_keep=1)
        ''' -- logging ---------------------------------------------------------- '''
        tensorboard_writer = TensorboardWriter(serialization_dir='tensorboard', summary_interval=1)
        if index == 0:
            shutil.rmtree(os.path.join(*['.', 'tensorboard', 'log']))

        optimizer = torch.optim.Adam(model.parameters(), lr=args_hpo.lr, weight_decay=args_hpo.l2)
        trainer = GradientDescentTrainer(
            model=model,
            optimizer=optimizer,
            data_loader=train_data_loader,
            validation_data_loader=val_data_loader,
            # note: this is the metric for early stopping
            validation_metric='-loss',
            patience=self.configurations.patience,
            num_epochs=self.configurations.max_epochs,
            serialization_dir=serialization_path_longer,
            checkpointer=model_checkpointer,
            cuda_device=self.configurations.device,
            grad_norm=args_hpo.clip,
            tensorboard_writer=tensorboard_writer,
            learning_rate_scheduler=ReduceOnPlateauLearningRateScheduler(optimizer=optimizer)
        )

        ''' trainer saves the model, but the vocabulary needs to be saved, too '''
        vocabulary.save_to_files(vocab_dir_longer)

        ''' check the metric names to synchronize with the class '''
        metrics = trainer.train()
        test_metrics = model.test(test_data=test_data, dataset_reader=dataset_reader)
        metrics.update(test_metrics)
        metrics['time_consumed(hrs)'] = round((time.time() - starting_time) / 3600, 4)

        return metrics
