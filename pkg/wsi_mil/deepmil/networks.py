"""
Implements the networks that can be used in train. 
use of pytorch.
"""
import functools
from torch.nn import (Linear, Module, Sequential, LeakyReLU, Tanh, Softmax, Identity, MaxPool2d, Conv3d,
                      Sigmoid, Conv1d, Conv2d, ReLU, Dropout, BatchNorm1d, BatchNorm2d, InstanceNorm1d, 
                      MaxPool3d, functional, LayerNorm, MultiheadAttention, LogSoftmax, ModuleList)
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
import copy
from torch.nn.parameter import Parameter
import torch
from torch.nn.init import (xavier_normal_, xavier_uniform_, constant_)
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import torchvision

def is_in_args(args, name, default):
    """Checks if the parammeter is specified in the args Namespace
    If not, attributes him the default value
    """
    if name in args:
        para = getattr(args, name)
    else:
        para = default
    return para

class MultiHeadAttention(Module):
    """
    Implements the multihead attention mechanism used in 
    MultiHeadedAttentionMIL_*. 
    Input (batch, nb_tiles, features)
    Output (batch, nb_tiles, nheads)
    """
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        width_fe = is_in_args(args, 'width_fe', 64)
        atn_dim = is_in_args(args, 'atn_dim', 256)
        self.num_heads = is_in_args(args, 'num_heads', 1)
        self.dropout = args.dropout
        self.dim_heads = atn_dim // self.num_heads
        assert self.dim_heads * self.num_heads == atn_dim, "atn_dim must be divisible by num_heads"

        self.atn_layer_1_weights = Parameter(torch.Tensor(atn_dim, args.feature_depth))
        self.atn_layer_2_weights = Parameter(torch.Tensor(1, 1, self.num_heads, self.dim_heads, 1))
        self.atn_layer_1_bias = Parameter(torch.empty((atn_dim)))
        self.atn_layer_2_bias = Parameter(torch.empty((1, self.num_heads, 1, 1)))
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.atn_layer_1_weights)
        xavier_uniform_(self.atn_layer_2_weights)
        constant_(self.atn_layer_1_bias, 0)
        constant_(self.atn_layer_2_bias, 0)

    def forward(self, x):
        """ Extracts a series of attention scores.

        Args:
            x (torch.Tensor): size (batch, nb_tiles, features)

        Returns:
            torch.Tensor: size (batch, nb_tiles, nb_heads)
        """
        bs, nbt, _ = x.shape

        # Weights extraction
        x = F.linear(x, weight=self.atn_layer_1_weights, bias=self.atn_layer_1_bias)
        x = torch.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view((bs, nbt, self.num_heads, 1, self.dim_heads))
        x = torch.matmul(x , self.atn_layer_2_weights) + self.atn_layer_2_bias # 4 scores.
        x = x.view(bs, nbt, -1) # shape (bs, nbt, nheads) 
        return x

class LinearBatchNorm(Module):
    def __init__(self, in_features, out_features, dropout, constant_size, dim_batch=None):
        if dim_batch is None:
            dim_batch = out_features
        super(LinearBatchNorm, self).__init__()
        self.cs = constant_size
        self.block = Sequential(
            Linear(in_features, out_features),
            ReLU(),# Added 25/09
            Dropout(p=dropout),# Added 25/09
            self.get_norm(constant_size, dim_batch),
                )
    def get_norm(self, constant_size, out_features):
        if not constant_size:
            norm = InstanceNorm1d(out_features)
        else:
            norm = BatchNorm1d(out_features)
        return norm

    def forward(self, x):
        x = self.block(x)
        return x

class MHMC_layers(Module):
    """
    MultiHeadMultiClass attention MIL, with several layers in the decision MLP.
    Same as MultiHeadedAttentionMIL_multiclass but have a classifier with N 
    Linear layers. 
    N is parametrized by args by args.n_layers_classif
    """
    def __init__(self, args):
        super(MHMC_layers, self).__init__()
        self.args = args
        self.dropout = args.dropout
        width_fe = is_in_args(args, 'width_fe', 64)
        atn_dim = is_in_args(args, 'atn_dim', 256)
        self.feature_depth = is_in_args(args, 'feature_depth', 512)
        self.num_heads = is_in_args(args, 'num_heads', 1)
        self.num_class = is_in_args(args, 'num_class', 2)
        self.n_layers_classif = is_in_args(args, 'n_layers_classif', 1)
        self.dim_heads = atn_dim // self.num_heads
        assert self.dim_heads * self.num_heads == atn_dim, "atn_dim must be divisible by num_heads"

        self.attention = Sequential(
            MultiHeadAttention(args),
            Softmax(dim=-2)
        )

        classifier = []
        classifier.append(LinearBatchNorm(int(args.feature_depth * self.num_heads), width_fe, args.dropout, args.constant_size))
        for i in range(self.n_layers_classif):
            classifier.append(LinearBatchNorm(width_fe, width_fe, args.dropout, args.constant_size))
        classifier.append(Linear(width_fe, self.num_class))
        classifier.append(LogSoftmax(-1))
        self.classifier = Sequential(*classifier)

    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patche
        """
        bs, nbt, _ = x.shape
        w = self.attention(x) # (bs, nbt, nheads)
        w = torch.transpose(w, -1, -2) # (bs, nheads, nbt)
        slide = torch.matmul(w, x) # Slide representation, shape (bs, nheads, nfeatures)
        slide = slide.flatten(1, -1) # (bs, nheads*nfeatures)
        if not self.args.constant_size:
            slide = slide.unsqueeze(-2)
        out = self.classifier(slide)
        out = out.view((bs, self.num_class))
        return out

class GeneralMIL(Module):
    """
    General MIL algorithm with tunable pooling function.
    Same as MultiHeadedAttentionMIL_multiclass but have a classifier with N 
    Linear layers. 
    N is parametrized by args by args.n_layers_classif
    """
    def __init__(self, args):
        super(GeneralMIL, self).__init__()
        ##  Set parameters
        self.dropout = args.dropout
        width_fe = is_in_args(args, 'width_fe', 64)
        atn_dim = is_in_args(args, 'atn_dim', 256)
        self.feature_depth = is_in_args(args, 'feature_depth', 512)
        self.num_heads = is_in_args(args, 'num_heads', 1)
        self.num_class = is_in_args(args, 'num_class', 2)
        self.n_layers_classif = is_in_args(args, 'n_layers_classif', 2)
        self.dim_heads = atn_dim // self.num_heads
        assert self.dim_heads * self.num_heads == atn_dim, "atn_dim must be divisible by num_heads"

        ## set networks
        self.args = copy.deepcopy(args)
        if self.args.instance_transf:
            self.instance_transf = Sequential(
                    LinearBatchNorm(int(args.feature_depth), 256, args.dropout,  args.constant_size), 
                    LinearBatchNorm(256, 128, args.dropout, args.constant_size))
            self.args.feature_depth = 128

        self.pooling_function = PoolingFunction(self.args)

        classifier = []
        dim_classif = self._get_first_dim(self.args)
        classifier.append(LinearBatchNorm(int(dim_classif), width_fe, self.args.dropout, self.args.constant_size))
        for i in range(self.n_layers_classif):
            classifier.append(LinearBatchNorm(width_fe, width_fe, self.args.dropout, self.args.constant_size))
        classifier.append(Linear(width_fe, self.num_class))
        classifier.append(LogSoftmax(-1))
        self.classifier = Sequential(*classifier)

    def _get_first_dim(self, args):
        if args.pooling_fct == 'conan':
            return 2 * args.k * args.feature_depth
        else:
            return args.feature_depth

    def forward(self, x):
        """
    Input x of size BxNxF where :
            * F is the dimension of feature space
            * N is number of patche
    """
        bs, nbt, _ = x.shape
        if self.args.instance_transf:
            x = x.view((bs*nbt, -1))
            x = self.instance_transf(x)
            x = x.view((bs, nbt, -1))
        slide = self.pooling_function(x)
        out = self.classifier(slide)
        out = out.view((bs, self.num_class))
        return out

class PoolingFunction(Module):
    def __init__(self, args):
        super(PoolingFunction, self).__init__()
        self.pooling = args.pooling_fct
        self.args = args
        self.k = args.k
        if self.pooling in ['ilse','max']:
            self.attention = Sequential(
                MultiHeadAttention(args),
                Softmax(dim=-2)
            )
        elif self.pooling in ['conan']:
             self.attention = Sequential(
                MultiHeadAttention(args)
            )
            

    def forward(self, x):
        if self.pooling == 'ilse':
            w = self.attention(x) # (bs, nbt, nheads)
            w = torch.transpose(w, -1, -2) # (bs, nheads, nbt)
            slide = torch.matmul(w, x) # Slide representation, shape (bs, nheads, nfeatures)
            slide = slide.flatten(1, -1) # (bs, nheads*nfeatures)

        elif self.pooling == 'mean':
            slide = torch.mean(x, -2) # BxF

        elif self.pooling == 'max_features':
            slide, _ = torch.max(x, dim=-2) # BxF

        elif self.pooling == 'max':
            w = self.attention(x)
            _, inds = torch.max(w, dim=-2)
            slide = torch.gather(x, 1, torch.cat([inds.unsqueeze(-1)] * self.args.feature_depth, axis=-1))
            slide = slide.squeeze(-2)

        elif self.pooling == 'conan':
            w = self.attention(x).squeeze(-1) # BxN
            _, topk = torch.topk(w, self.k, -1)
            _, lowk = torch.topk(-w, self.k, -1)
            topk = torch.gather(x, 1, torch.cat([topk.unsqueeze(-1)] * self.args.feature_depth, axis=-1))
            lowk = torch.gather(x, 1, torch.cat([lowk.unsqueeze(-1)] * self.args.feature_depth, axis=-1))
            slide = torch.cat([topk, lowk], axis=1) # Bx2kxF
            slide = slide.flatten(-2,-1)

        else:
            print(f'{self.pooling} pooling function not yet implemented``')
        return slide

class MultiHeadedAttentionMIL(Module):
    """
    Implements deepMIL while taking 1D vectors as instances (output of resnet for example)
    Bags are then NxF matrices, F the feature number (usually 2048), N the number of instances of the bag.
    Use with BCE loss.
    """
    def __init__(self, args):
        super(MultiHeadedAttentionMIL, self).__init__()
        self.dropout = args.dropout
        width_fe = is_in_args(args, 'width_fe', 64)
        atn_dim = is_in_args(args, 'atn_dim', 256)
        self.num_heads = is_in_args(args, 'num_heads', 1)
        self.feature_depth = is_in_args(args, 'feature_depth', 512)
        self.dim_heads = atn_dim // self.num_heads
        assert self.dim_heads * self.num_heads == atn_dim, "atn_dim must be divisible by num_heads"

        self.attention = Sequential(
            MultiHeadAttention(args)
        )

        self.classifier = Sequential(
            Linear(int(args.feature_depth * self.num_heads), width_fe),
            ReLU(),# Added 25/09
            Dropout(p=args.dropout),# Added 25/09
            Linear(width_fe, width_fe),
            ReLU(),# Added 25/09
            Dropout(p=args.dropout),# Added 25/09
            Linear(width_fe, 1),# Added 25/09
            Sigmoid()
        )

    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patche
        """
        bs, nbt, _ = x.shape #x : (bs, nbt, nfeatures)
        w = self.attention(x) # (bs, nbt, nheads)
        w = F.softmax(w, dim=-2)
        w = torch.transpose(w, -1, -2) # (bs, nheads, nbt)
        slide = torch.matmul(w, x) # Slide representation, shape (bs, nheads, nfeatures)
        slide = slide.flatten(1, -1) # (bs, nheads*nfeatures)
        out = self.classifier(slide)
        out = out.view(bs)
        return out

def get_norm_layer(use_bn=True, d=1):
    bn_dict = {1:BatchNorm1d, 2:BatchNorm2d}
    if use_bn: #Use batch
        norm_layer = functools.partial(bn_dict[d], affine=True, track_running_stats=True)
    else:
        norm_layer = functools.partial(Identity)
    return norm_layer

class model1S(Module):
    """
    Args must contain : 
        * feature_depth: int, number of features of the inuput
        * dropout: float, dropout parameter
    
    Ends witha Softmax, to use BCELoss 
    Takes as input a WSI as a Tensor of shape BxNxD where :
        * D the feature depth
        * N the number of tiles
        * B the batch dimension

    The first operation is to transform the tensor in the form (B)xDxN 
    so that D is the 'channel' dimension
    Use with BCELoss.
    """
    def __init__(self, args):
        super(model1S, self).__init__()
        use_bn = args.constant_size & (args.batch_size > 8)
        norm_layer = get_norm_layer(use_bn)
        n_clusters = is_in_args(args, 'n_clusters', 128)
        hidden_fcn = is_in_args(args, 'hidden_fcn', 64)
        self.continuous_clusters = Sequential(
            Conv1d(in_channels=args.feature_depth, 
                   out_channels=n_clusters,
                   kernel_size=1),
            norm_layer(n_clusters),
            ReLU(),
            Dropout(p=args.dropout)
        )
        self.classifier = Sequential(
            Linear(in_features=n_clusters,
                   out_features=hidden_fcn), # Hidden_fc
            norm_layer(hidden_fcn),
            ReLU(),
            Dropout(p=args.dropout),
            Linear(in_features=hidden_fcn,
                   out_features=1),
            Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.continuous_clusters(x)
        out = out.max(dim=-1)[0] # AvgPooling
        out = self.classifier(out)
        out = out.squeeze(-1)
        return out

class Conv2d_bn(Module):
    def __init__(self, in_channels, out_channels, dropout, use_bn):
        super(Conv2d_bn, self).__init__()
        self.norm_layer = get_norm_layer(use_bn, d=2)
        self.layer = Sequential(
            Conv2d(in_channels=in_channels, 
                   out_channels=out_channels,
                   kernel_size=(3, 3),
                   padding=(1, 1)),
            self.norm_layer(out_channels),
            ReLU(),
            Dropout(p=dropout)
        )
    def forward(self, x):
        out = self.layer(x)
        return out

class Conv1d_bn(Module):
    def __init__(self, in_channels, out_channels, dropout, use_bn):
        self.norm_layer = get_norm_layer(use_bn)
        super(Conv1d_bn, self).__init__()
        self.layer = Sequential(
            Conv1d(in_channels=in_channels, 
                   out_channels=out_channels,
                   kernel_size=1),
            self.norm_layer(out_channels),
            ReLU(),
            Dropout(p=dropout)
        )
    def forward(self, x):
        out = self.layer(x)
        return out

class Dense_bn(Module):
    def __init__(self, in_channels, out_channels, dropout, use_bn):
        self.norm_layer = get_norm_layer(use_bn)
        super(Dense_bn, self).__init__()
        self.layer = Sequential(
            Linear(in_features=in_channels, 
                   out_features=out_channels),
            self.norm_layer(out_channels),
            ReLU(),
            Dropout(p=dropout)
        )
    def forward(self, x):
        out = self.layer(x)
        return out

class Conan(Module):
    """
    Args must contain : 
        * feature_depth: int, number of features of the inuput
        * dropout: float, dropout parameter
    
    Ends witha Softmax, to use BCELoss 
    Takes as input a WSI as a Tensor of shape BxNxD where :
        * D the feature depth
        * N the number of tiles
        * B the batch dimension

    The first operation is to transform the tensor in the form (B)xDxN 
    so that D is the 'channel' dimension

    """
    def __init__(self, args):
        self.k = 5
        self.hidden1 = is_in_args(args, 'hidden1', 256)
        self.hidden2 = self.hidden1//4 
        self.hidden_fcn = is_in_args(args, 'hidden_fcn', 128)
        use_bn = args.constant_size & (args.batch_size > 8)
        super(Conan, self).__init__()
        self.continuous_clusters = Sequential(
            Conv1d_bn(in_channels=args.feature_depth,
                      out_channels=self.hidden1, 
                      dropout=args.dropout, 
                      use_bn=use_bn),
            Conv1d_bn(in_channels=self.hidden1,
                      out_channels=self.hidden2, 
                      dropout=args.dropout, 
                      use_bn=use_bn),
            Conv1d_bn(in_channels=self.hidden2,
                      out_channels=self.hidden1, 
                      dropout=args.dropout,
                      use_bn=use_bn),
        )
        self.weights = Sequential(
            Conv1d(in_channels=self.hidden1, 
                   out_channels=1,
                   kernel_size=1),
            ReLU()
        )
        self.classifier = Sequential(
            Dense_bn(in_channels=(self.hidden1 + 1) * 2 * self.k + self.hidden1,
                     out_channels=self.hidden_fcn,
                     dropout=args.dropout, 
                     use_bn=use_bn),
            Dense_bn(in_channels=self.hidden_fcn,
                     out_channels=self.hidden_fcn, 
                     dropout=args.dropout,
                     use_bn=use_bn),
            Linear(in_features=self.hidden_fcn,
                   out_features=args.num_class),
            LogSoftmax(-1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.continuous_clusters(x)
        scores = self.weights(out)
        _, indices = torch.sort(scores, dim=-1)

        ## Aggregation
        selection = torch.cat((indices[:, :, :self.k], indices[:, :, -self.k:]), axis=-1)
        selection_out = torch.cat([selection] *self.hidden1, axis=1)
        out = torch.gather(out, -1, selection_out)
        scores = torch.gather(scores, -1, selection)
        avg = torch.mean(out, dim=-1)
        out = torch.cat((scores.flatten(1, -1), avg.flatten(1, -1), out.flatten(1, -1)), axis=-1)
        out = self.classifier(out)
        return out
    
class SelfAttentionMIL(Module):
    """Implemented according to Li et Eliceiri 2020
    Use with BCELoss.
    """
    def __init__(self, args):
        super(SelfAttentionMIL, self).__init__()
        self.L = 128
        self.args = args
        self.maxmil = Sequential(
            Linear(args.feature_depth, 256),
            ReLU(),
            Dropout(args.dropout),
            Linear(int(2*self.L), self.L),
            ReLU(),
            Dropout(args.dropout),
            Linear(self.L, 1)
            )
        self.queries = Sequential(
            Linear(args.feature_depth, int(2*self.L)),
            ReLU(),
            Dropout(args.dropout),
            Linear(int(self.L*2), self.L)
            )
        self.visu = Sequential(
            Linear(args.feature_depth, int(self.L*2)),
            ReLU(),
            Dropout(args.dropout),
            Linear(int(self.L*2), self.L)
            )
        self.wsi_score = Sequential(
            Linear(self.L, 1)
            )
        self.classifier = Sequential(Linear(2, 1), Sigmoid())

    def forward(self, x):
        # Ingredients of the self attention
        milscores = self.maxmil(x)
        queries = self.queries(x)
        visu = self.visu(x)

        max_scores, max_indices = torch.max(milscores, dim=1)
        max_scores = max_scores.unsqueeze(-1)
        max_indices = torch.cat([max_indices] * self.L, axis=-1).unsqueeze(1) 
        max_query = torch.gather(queries, -2, max_indices)
        max_query = max_query.permute(0, 2, 1)
        sa_scores = torch.matmul(queries, max_query)
        sa_scores = sa_scores.permute(0, 2, 1)
        sa_scores = functional.softmax(sa_scores, dim=-1)
        weighted_visu = torch.matmul(sa_scores, visu)
        wsi_scores = self.wsi_score(weighted_visu)
        fused = torch.cat([max_scores, wsi_scores], axis=-2).squeeze(-1)
        x = self.classifier(fused)
        x = x.squeeze(-1)
        return x

class TransformerMIL(Module):
    def __init__(self, args):
        super(TransformerMIL, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=args.feature_depth, nhead=8, dim_feedforward=2048, dropout=args.dropout, activation="relu")
        encoder_norm = LayerNorm(args.feature_depth)
        self.attention = TransformerEncoder(encoder_layer, args.ntrans, encoder_norm)
        self.attention2 = MultiheadAttention(args.feature_depth, 8)
        self.classifier = Sequential(
            Linear(args.feature_depth, 1),
            Sigmoid())
        self.mil = MHMC_layers(args)
    def forward(self, x):
        x, _ = self.attention2(x, x, x)
        x = self.mil(x)
        return x

class MILGene(Module):
    """MILGene.
    MIL generator. This framework makes things easy if we want to plug the 
    MIL framework on a learnable feature extractor (taking images as input).
    Not implemented here so.
    """
    models = { 'generalmil': GeneralMIL, 
                'multiheadmil': MultiHeadedAttentionMIL,
                'mhmc_layers': MHMC_layers,
                'conan': Conan, 
                '1s': model1S, 
                'sa': SelfAttentionMIL,
                'transformermil': TransformerMIL
                }     

    def __init__(self, args):
        super(MILGene, self).__init__()
        self.args = args
        self.features_tiles = Identity(args)
        self.mil = self.models[args.model_name](args)

    def forward(self, x):
        if self.args.constant_size:
            batch_size, nb_tiles = x.shape[0], x.shape[1]
        else:
            batch_size, nb_tiles = 1, x.shape[-2]
        x = self.features_tiles(x)
        print(x.shape)
        x = x.view(batch_size, nb_tiles, self.args.feature_depth)
        x = self.mil(x)
        return x
