"""
utils to facilitate AllenNLP based training
"""
from allennlp.training.callbacks import Callback, handle_event, Events
from torch.utils.tensorboard import SummaryWriter


class AllenNLPTensorboardLogger(Callback):
    """
        log metrics, etc. to tensorboard with torch.utils.tensorboard for AllenNLP models.
    """

    def __init__(self, log_folder, metric_s, input_to_model, log_embeddings=True, add_graph=False):
        """

        :param log_folder: full path of log folder corresponding to the CURRENT execution.
        :param metric_s: metrics to log. a subset of the keys of trainer.train_metrics and/or trainer.val_metrics.
        :param input_to_model: a sample instance as input to the model, required by tensorboard.
        :param log_embeddings: bool. If True, the callback calls
                trainer.model.prepare_embeddings_for_tensorboard()
            to access weights. The function is supposed to return a list of dict each with
            'mat': V * E
            'metadata': V
            'tag': the namespace of the embeddings, e.g. 'words'.
        """
        self.writer = SummaryWriter(log_folder)
        self.metric_s = metric_s
        self.log_embeddings = log_embeddings
        self.input_to_model = input_to_model
        self.add_graph = add_graph
        super().__init__()

    @handle_event(Events.TRAINING_START)
    def log_graph(self, trainer):
        ''' fixme: yes this is fishy, it seems input examples are mandatory. '''
        if self.add_graph:
            self.writer.add_graph(trainer.model, input_to_model=self.input_to_model)

    @handle_event(Events.TRAINING_END)
    def log_hparams(self, trainer):
        self.writer.add_hparams(
            hparam_dict=trainer.model.args_hpo.__dict__,
            metric_dict={'hp_val_accuracy': trainer.metrics['best_validation_accuracy'],
                         'hp_val_loss': trainer.metrics['best_validation_loss'],
                         'hp_train_accuracy': trainer.metrics['training_accuracy'],
                         'hp_train_loss': trainer.metrics['training_loss']})

    @handle_event(Events.EPOCH_END)
    def log_metrics(self, trainer):
        if self.metric_s:
            for metric in self.metric_s:
                for phase in ['train', 'val']:
                    self.writer.add_scalar(tag='{}_{}'.format(phase, metric),
                                           scalar_value=getattr(trainer, '{}_metrics'.format(phase))[metric],
                                           global_step=trainer.epoch_number)

    @handle_event(Events.EPOCH_END)
    def log_embedding(self, trainer):
        if self.log_embeddings:
            embedding_data = trainer.model.prepare_embeddings_for_tensorboard()
            for embedder in embedding_data:
                self.writer.add_embedding(mat=embedder['mat'],
                                          metadata=embedder['metadata'],
                                          tag=embedder['tag'],
                                          global_step=trainer.epoch_number)

    @handle_event(Events.TRAINING_END)
    def close_writer(self, trainer):
        self.writer.close()
