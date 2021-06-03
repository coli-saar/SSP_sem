import dill
import os
import datetime

import numpy as np

from misc import Struct, PrintColors


class RandomSearchMetaOptimizer:
    """
    to perform random hyper-parameter search
    usage:
        inherit the class and override self.train()
        call self.search to perform the search and log results
    """

    def __init__(self, parameters: dict, num_trials: int, tag: str, metric_names: list):
        """
        generates parameter combinations for random parameter search, stored in self.combinations as dictionaries
        all sampling parameters are dictionaries formed as dict{hyper_name: value}

        :param parameters:
            the parameters involved in hyper parameter search. a dict of dicts. The first hierarchy of keys are the
            parameter names; the second hierachy should include the following:
            'domain': the range of the parameter. should be a tuple.
                Note: for exponential sampling, the upperbound of the domain does not get sampled.
            'sample_criterion':
                'u': uniform over the domain
                '2e': the parameter's logrithm wrt 2 is sampled uniformly as an INTEGER
                '10e': the parameter's logrithm is sampled uniformly as a FLOAT
            'type':
                'int': floor and returns an integer
                'float': float, direct copy
        :param metric_names:
            the name of the metrics returned by .train() that should be logged. if allennlp is used, the metric names
            are usually expected from trainer.train(), i.e. trainer.metrics.
                NOTE: for AllenNLP, metric names need to be prefixed with 'vest_validation ... ' to record the metrics
                    out of the best model. Otherwise the metrics of the last model will be recorded.
        :param num_trials:
        :param tag:
        """
        self.hyper_combs = [Struct() for _ in range(num_trials)]
        self.num_trials = num_trials
        self.tag = tag
        self.log_path = f"logs_{self.tag}_{datetime.date.today()}.csv"
        self.combs_path = "hyper_combs_{}".format(self.tag)
        self.parameters = parameters
        self.hyper_names = [hyper for hyper in self.parameters]
        self.metric_names = metric_names

        ''' generate parameters '''
        for i, combination in enumerate(self.hyper_combs):
            for hyper in self.parameters:
                # return if only a single quantity is given
                if not isinstance(self.parameters[hyper]['domain'], list):
                    assert isinstance(self.parameters[hyper]['domain'], int) \
                           or isinstance(self.parameters[hyper]['domain'], float)
                    combination[hyper] = self.parameters[hyper]['domain']
                    continue
                else:
                    min_value, max_value = self.parameters[hyper]['domain']
                    assert min_value <= max_value
                rnd_ready = None
                if self.parameters[hyper]['sample_criterion'] == '2e':
                    assert min_value > 0
                    min_exp, max_exp = np.log2(min_value), np.log2(max_value)
                    rnd = np.random.uniform() * (max_exp - min_exp) + min_exp
                    rnd_ready = np.power(2., np.floor(rnd))
                elif self.parameters[hyper]['sample_criterion'] == '10e':
                    assert min_value > 0
                    min_exp, max_exp = np.log10(min_value), np.log10(max_value)
                    rnd = np.random.uniform() * (max_exp - min_exp) + min_exp
                    rnd_ready = np.power(10., rnd)
                elif self.parameters[hyper]['sample_criterion'] == 'u':
                    rnd_ready = np.random.uniform() * (max_value - min_value) + min_value

                if self.parameters[hyper]['type'] == 'int':
                    combination[hyper] = int(rnd_ready)
                elif self.parameters[hyper]['type'] == 'float':
                    combination[hyper] = rnd_ready

        ''' initialize log if applicable '''
        if not os.path.exists(self.log_path):
            header = 'index' + ',' + ','.join(self.hyper_names) + ',' + ','.join(self.metric_names)
            with open(self.log_path, 'w') as log:
                log.write(header + '\n')

        ''' save combinations '''
        dill.dump(self.hyper_combs, open(self.combs_path, 'wb'))

    def _perform_search(self, hyper_comb, execution_idx):
        print('------ Random Hyper Search Round {} ------'.format(execution_idx))
        metrics = self.train(hyper_comb, execution_idx)
        log_line = str(execution_idx) + ',' + \
            ','.join(['{:.3g}'.format(hyper_comb[name]) for name in self.hyper_names]) + ',' + \
            ','.join(['{:.3g}'.format(metrics[name]) for name in self.metric_names])
        with open(self.log_path, 'a') as log_out:
            log_out.write(log_line + '\n')

    def search(self, test_mode=True):
        """
        main entrance, execute to perform the optimization and log the parameters.
        :param test_mode:
        :return:
        """
        PrintColors.prRed(f'======Performing Random Hyper Search for execution {self.tag}======')
        for execution_idx, hyper_comb in enumerate(self.hyper_combs):
            # self._perform_search(hyper_comb, execution_idx)
            if test_mode:
                self._perform_search(hyper_comb, execution_idx)
            else:
                try:
                    self._perform_search(hyper_comb, execution_idx)
                except RuntimeError as rte:
                    PrintColors.prPurple(rte)
                    continue

    def train(self, combination, index):
        """
        override to execute one round of random hyper-parameter search, and returns the metrics that needs to be logged.

        :param combination: hyper parameter combination
        :param index
        :return: metrics for evaluation as a dictionary
        """
        raise NotImplementedError




