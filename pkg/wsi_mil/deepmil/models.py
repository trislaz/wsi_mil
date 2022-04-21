"""
implementing models. DeepMIL implements a models that classify a whole slide image
"""


from torch.nn import BCELoss, NLLLoss, MSELoss, LogSoftmax
from torch.optim import Adam
import torch
import numpy as np
from sklearn import metrics
import torch 
import torchvision
from abc import ABC, abstractmethod
#from torch.utils.tensorboard import SummaryWriter
import shutil
import os
from .networks import MILGene
from .dataloader import Dataset_handler

class Model(ABC):
    def __init__(self, args):
        self.args = args
        self.optimizers = []
        self.losses = {'global':[]}
        self.metric = 0
        self.criterion = lambda : 1
        self.counter = {'epoch': 0, 'batch': 0}
        self.network = torch.nn.Module()
        self.early_stopping = EarlyStopping(args=args)
        self.device = args.device
        self.ref_metric = args.ref_metric
        self.best_metrics = None
        self.best_ref_metric = None
        self.dataset = None
        self.writer = None

    @abstractmethod
    def optimize_parameters(self, input_batch, target_batch):
        pass

    @abstractmethod
    def make_state(self):
        pass

    @abstractmethod
    def predict(self, x):
        """Makes a prediction about the label of x.
        Prediction should be in numpy format.
        
        Parameters
        ----------
        x : torch.Tensor
            input
        """
        pass

    @abstractmethod
    def evaluate(self, x, y):
        pass

    def get_summary_writer(self):
        if 'EVENTS_TF_FOLDER' in os.environ:
            directory = os.environ['EVENTS_TF_FOLDER']
        else:
            directory = None
        self.writer = SummaryWriter(directory)

    def update_learning_rate(self, metrice):
        for sch in self.schedulers:
            sch.step(metric)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def set_zero_grad(self):
        optimizers = self.optimizers
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            if optimizer is not None:
                optimizer.zero_grad()

class EarlyStopping:
    """Early stopping AND saver !
    """
    def __init__(self, args):
        self.patience = args.patience
        self.counter = 0
        self.loss_min = None
        self.early_stop = False
        self.is_best = False
        self.filename = 'model.pt.tar'

    def __call__(self, loss, state):
        if self.loss_min is None:
            self.loss_min = loss
            self.is_best = True
        elif self.loss_min <= loss :
            self.counter += 1
            self.is_best = False
            if self.counter < self.patience:
                if False:
                    print('-- There has been {}/{} epochs without improvement on the validation set. --\n'.format(
                          self.counter, self.patience))
            else:
                self.early_stop = True
        else:
            print(loss, self.loss_min, ' early stop!')
            self.is_best = True
            self.loss_min = loss
            self.counter = 0
        self.save_checkpoint(state)

    def save_checkpoint(self, state):
        torch.save(state, self.filename)
        if self.is_best:
            print('SAVING BEST')
            shutil.copyfile(self.filename, self.filename.replace('.pt.tar', '_best.pt.tar'))     
            
class DeepMIL(Model):
    """
    Class implementing a Deep-MIL framwork, for WSI classification. 
    """

    def __init__(self, args, label_encoder=None, with_data=False, ipca=None):
        """
        args contains all the info for initializing the deepmil and its dataloader.

        :param args: Namespace. outputs of .arguments.get_arguments.
        :param label_encoder: sklearn.LabelEncoder, default = None.
        :param with_data: bool, when True : set the data_loaders according to args. When False, 
        The MIL model is loader without the data. 
        Order of calls for _constructors fucntions is important.
        """
        super(DeepMIL, self).__init__(args)
        self.results_val = {'proba_preds': [],
                            'y_true': [],
                            'preds':[], 
                            'scores': []}
        self.scores_dpp = []
        self.mean_train_loss = 0
        self.mean_val_loss = 0
        self.model_name = args.model_name
        self.network = self._get_network()
        self.network.num_class = args.num_class
        optimizer = self._get_optimizer(args)
        self.optimizers = [optimizer]
        self.schedulers = self._get_schedulers(args)
        self.train_loader, self.val_loader = self._get_data_loaders(args, with_data)
        self.label_encoder = self.train_loader.dataset.label_encoder if label_encoder is None else label_encoder
        # when training ipca = None, when predicting, ipca is given when loading.
#        self.ipca = self.train_loader.dataset.ipca if ipca is None else ipca 
        self.ipca =  ipca 
        self.criterion = self._get_criterion(args.criterion)
        self.bayes = False

    def _get_network(self):
        """_get_network.
        Initialize the network and transfer it on the cuda device.

        :return nn.Module: MIL network.
        """
        if self.args.model_path is not None and self.args.model_name != 'sparseconvmil':
            ## TODO add that part INSIDE the network initialisation....
            ckpt = torch.load(self.args.model_path, map_location='cpu')
            sd = ckpt['state_dict']
            net = MILGene(self.args)#ckpt['args_mil'])       
            for k in list(sd.keys()):
                if k.startswith('module.encoder.'):
                    sd[k[len('module.encoder.'):]] = sd[k]
                del sd[k]
            msg = net.load_state_dict(sd, strict=False)
            net.mil.classifier.add_module(module=torch.nn.Linear(512, self.args.num_class), name='fc')
            net.mil.num_class = self.args.num_class 
            #Because num_class for a MIL-SSL is the representation dimension (512) have to change it to the final fc dim = real number of class of the downstream task
        else:
            net = MILGene(args=self.args)
        net = net.to(self.args.device)
        return net
    
    def _get_data_loaders(self, args, with_data):
        """_get_data_loaders.

        Gets the loaders for training.

        :param args: Namespace. Outputs of .arguments.get_arguments.
        :param with_data: bool, if TRUE, loads data, else not.
        """
        train_loader, val_loader, label_encoder = None, None, None
        if with_data:
            data = Dataset_handler(args)
            train_loader, val_loader = data.get_loader(training=True)
        return train_loader, val_loader

            
    def _get_schedulers(self, args):
        """_get_schedulers.
        Must be called after having define the optimizers (self._get_optimizer()) 
        
        Get the learning rate scheduler for the optimizer.
        :param args: Namespace. Outputs of .arguments.get_arguments.
        """
        if args.lr_scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            schedulers = [scheduler(optimizer=o, patience=self.args.patience_lr, factor=0.3) for o in self.optimizers]
        if args.lr_scheduler == 'cos':
            schedulers = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=o, T_0=1, T_mult=2) for o in self.optimizers]
        return schedulers

    def _get_optimizer(self, args):
        """_get_optimizer.

        :param args: Namespace. Outputs of .arguments.get_arguments.
        """
        if args.optimizer == 'adam':
            optimizer = Adam(self.network.parameters(), lr=args.lr)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.network.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)
        return optimizer

    def _get_criterion(self, criterion):
        """_get_criterion.
        Constructor of the loss function

        :param criterion: str, loss_function
        """
        if criterion == 'bce':
            criterion = BCELoss().to(self.args.device)
        if criterion == 'nll': # to use with attentionmilmultihead.
            criterion = NLLLoss().to(self.args.device)
        return criterion
    
    def _forward_no_grad(self, x, xy=None):
        """_forward_no_grad.

        :param x: toch.tensor. 
        """
        with torch.no_grad():
            if xy is None:
                out = self.network(x)
            else:
                out = self.network((x, xy))
            out = LogSoftmax(dim=-1)(out)
        out = out.detach()
        return out

    def _to_pseudo_proba(self, out):
        """_to_pseudo_proba.

        When LogSoftmax is used as the final layer of the network, 
        we need to exp the output to get the pseudo-probas.

        :param out: torch.tensor or ndarray, output of the MIL network
        :return type(out), pseudo proba.
        """
        if self.model_name in ['multiheadmulticlass', 'mhmc_conan', 'mhmc_layers', 'generalmil', 'sparseconvmil']:
            return np.exp(out)
        else:
            return out

    def _get_pred_pseudo_proba(self, scores):
        """_get_pred_pseudo_proba.
        
        Get the maximum (=prediction) pseudo-proba.
        :param scores: torch.tensor, outputs of the MIL network. shape: NxC, C the number of classes.
        :return ndarray, best_pseudoproba, shape Nx1
        """
        proba = self._to_pseudo_proba(scores).numpy().max(axis=-1)
        return proba

    def _keep_best_metrics(self, metrics):
        """_keep_best_metrics.

        Stores the val metrics if this iteration is the best one, according to 
        the ref_metric.
        :param metrics: dict of the validation metrics of the current epoch.
        """
        factor = self.args.sgn_metric 
        if self.best_ref_metric is None:
            self.best_ref_metric = metrics[self.ref_metric]
            self.best_metrics = metrics
        if self.best_ref_metric * factor > metrics[self.ref_metric] * factor:
            print('old acc : {}, new acc : {}'.format(self.best_ref_metric, metrics[self.ref_metric]))
            self.best_ref_metric = metrics[self.ref_metric]
            self.best_metrics = metrics
        
    def flush_val_metrics(self):
        """flush_val_metrics.

        Once the forward pass for validation ends, computes the metrics, stores
        the best ones according to the ref metric.
        Metrics computed : Balanced accuracy, accuracy, precision (macro-avg), 
        recall, f1-score, roc_auc (when N_classes < 3), epochs, mean_train_loss, 
        mean_val_loss.

        :return dict, key (name of the metric), value (value of the metric).
        """
        val_scores = np.array(self.results_val['scores'])
        val_y = np.array(self.results_val['y_true'])
        val_metrics = self._compute_metrics(scores=val_scores, y_true=val_y)
        val_metrics['mean_train_loss'] = self.mean_train_loss
        val_metrics['mean_val_loss'] = self.mean_val_loss
        self._keep_best_metrics(val_metrics)

        # Re Initialize val_results for next validation
        self.results_val['scores'] = []
        self.results_val['y_true'] = []
        self.results_val['proba_preds'] = []

        return val_metrics

    def _predict_function(self, scores):
        """
        depends on the framework.
        Here, using BCE or NLL loss, always using argmax as the prediction.
        """
        preds = np.argmax(scores, axis=-1)
        return preds

    def _compute_metrics(self, scores, y_true):
        """_compute_metrics.
        Metrics computed : Balanced accuracy, accuracy, precision (macro-avg), 
        recall, f1-score, roc_auc (when N_classes < 3), epochs.

        :param scores: validation set output of the MIL network.
        :param y_true: labels of this validation set.
        """
        report = metrics.classification_report(y_true=y_true, y_pred=self._predict_function(scores), output_dict=True, zero_division=0)
        ba = metrics.balanced_accuracy_score(y_true=y_true, y_pred=self._predict_function(scores))
        metrics_dict = {'balanced_acc': ba, 'accuracy': report['accuracy'], "precision": report['macro avg']['precision'], 
            "recall": report['macro avg']['recall'], "f1-score": report['macro avg']['f1-score']}
        if self.args.num_class <= 2:
            metrics_dict['roc_auc'] = metrics.roc_auc_score(y_true=y_true, y_score=scores[:,1])
        metrics_dict['epoch'] = self.counter['epoch']
        return metrics_dict

    def predict(self, x):
        """predict.
        
        :param x: torch.tensor shape 1x1xF, F=number of features
        :return pseudo_probas (nxC), pred
        """
        x = x.to(self.args.device)
        proba = self._to_pseudo_proba(self._forward_no_grad(x).cpu())
        pred = int(self._predict_function(proba).item())
        pred = self.label_encoder.inverse_transform([pred]).item()
        return proba.numpy(), pred

    def evaluate_kdpp(self, x, y, end, xy=None):
        """
        takes x, y torch.Tensors.
        Predicts on x, stores y and the loss, and the outputs of the network.
        n: number of iteration of sampling (and evaluation)
        """
        y = y.to(self.args.device, dtype=torch.int64)
        x = x.to(self.args.device)
        score = self._forward_no_grad(x, xy).to('cpu')
        self.scores_dpp.append(score)
        loss = self.criterion(score, y.to('cpu'))       
        if end:
            scores = torch.cat(self.scores_dpp).mean(0)
            print(scores)
            y = y.to('cpu', dtype=torch.int64)
            pred = int(self._predict_function(self._to_pseudo_proba(scores)).item())
            pred = self.label_encoder.inverse_transform([pred])
            proba = self._get_pred_pseudo_proba(scores)
            self.results_val['scores'].append(self._to_pseudo_proba(scores.numpy()))
            self.results_val['proba_preds'] += [proba.item()]
            self.results_val['y_true'] += list(y.cpu().numpy())
            self.results_val['preds'] += [pred.item()]
            self.scores_dpp = []
        return loss.detach().cpu().item()

    def evaluate(self, x, y, xy=None):
        """
        takes x, y torch.Tensors.
        Predicts on x, stores y and the loss, and the outputs of the network.
        """
        y = y.to(self.args.device, dtype=torch.int64)
        x = x.to(self.args.device)
        scores = self._forward_no_grad(x, xy)
        y = y.to('cpu', dtype=torch.int64)
        scores = scores.to('cpu')
        pred = int(self._predict_function(self._to_pseudo_proba(scores)).item())
        pred = self.label_encoder.inverse_transform([pred])
        proba = self._get_pred_pseudo_proba(scores)
        loss = self.criterion(scores, y)       
        self.results_val['scores'] += list(self._to_pseudo_proba(scores.numpy()))
        self.results_val['proba_preds'] += [proba.item()]
        self.results_val['y_true'] += list(y.cpu().numpy())
        self.results_val['preds'] += [pred]
        return loss.detach().cpu().item()

    def forward(self, x, xy=None):
        if xy is not None:
            out = self.network((x, xy))
        else:
            out = self.network(x)
        return out

    def optimize_parameters(self, input_batch, target_batch, xy=None):
        """optimize_parameters.

        Feed the network with a batch and optimize the parameter.
        Dataloader iterates (X, y), X being the data, y the label
        :param input_batch: data, X
        :param target_batch: label of X, y
        """
        self.set_zero_grad()
        if self.args.constant_size: # We can process a batch as a whole big tensor
            input_batch = input_batch.to(self.args.device)
            target_batch = target_batch.to(self.args.device, dtype=torch.int64)
            output = self.forward(input_batch, xy)
            output = LogSoftmax(dim=-1)(output)
            loss = self.criterion(output, target_batch)
            print(loss)
            loss.backward()

        else: # We have to process a batch as a list of tensors (of different sizes)
            loss = 0
            for o, im in enumerate(input_batch):
                im = im.to(self.args.device)
                target = target_batch[o].to(self.args.device, dtype=torch.int64)
                output = self.forward(im, xy)
                loss += self.criterion(output, target)
            loss = loss/len(input_batch)
            loss.backward()

        self.optimizers[0].step()
        return loss.detach().cpu().item()

    def make_state(self):
        """make_state.
        Creates a dictionnary checkpoint of the model.
        """
        dictio = {'state_dict': self.network.state_dict(),
                'state_dict_optimizer': self.optimizers[0].state_dict, 
                'state_scheduler': self.schedulers[0].state_dict(), 
                'inner_counter': self.counter,
                'args': self.args,
                'table_data': self.train_loader.dataset.table_data,
                'files_train': self.train_loader.dataset.files,
                'best_metrics': self.best_metrics, 
                'label_encoder': self.train_loader.dataset.label_encoder
                #'ipca': self.ipca
                }
        return dictio
