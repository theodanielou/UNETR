from nnunet.network_architecture.Transformer_architecture import *

from typing import Sequence, Tuple, Union


from nnunet.network_architecture.custom_modules.conv_blocks import StackedConvLayers, DeconvConvDropoutNormReLU, DecodeurLayer, ConvDropoutNormReLU
from nnunet.network_architecture.generic_UNet import Upsample
from nnunet.network_architecture.neural_network import SegmentationNetwork
import numpy as np

import torch.nn as nn
from monai.utils import ensure_tuple_rep

def get_default_network_config(dim=3, dropout_p=None, nonlin="LeakyReLU", norm_type="bn"):
    """
    returns a dictionary that contains pointers to conv, nonlin and norm ops and the default kwargs I like to use
    :return:
    """
    props = {}
    if dim == 2:
        props['conv_op'] = nn.Conv2d
        props['dropout_op'] = nn.Dropout2d
        props['deconv_op'] = nn.ConvTranspose2d
    elif dim == 3:
        props['conv_op'] = nn.Conv3d
        props['dropout_op'] = nn.Dropout3d
        props['deconv_op'] = nn.ConvTranspose3d
    else:
        raise NotImplementedError

    if norm_type == "bn":
        if dim == 2:
            props['norm_op'] = nn.BatchNorm2d
        elif dim == 3:
            props['norm_op'] = nn.BatchNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
    elif norm_type == "in":
        if dim == 2:
            props['norm_op'] = nn.InstanceNorm2d
        elif dim == 3:
            props['norm_op'] = nn.InstanceNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
    
    else:
        raise NotImplementedError

    if dropout_p is None:
        props['dropout_op'] = None
        props['dropout_op_kwargs'] = {'p': 0, 'inplace': True}
    else:
        props['dropout_op_kwargs'] = {'p': dropout_p, 'inplace': True}

    props['conv_op_kwargs'] = {'stride': 1, 'dilation': 1, 'bias': False}  # kernel size will be set by network!
    props['deconv_op_kwargs'] = {'stride': 2, 'dilation': 1, 'bias': False}  #UNETR ne semble pas utiliser de dilatation

    if nonlin == "LeakyReLU":
        props['nonlin'] = nn.LeakyReLU
        props['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
    elif nonlin == "ReLU":
        props['nonlin'] = nn.ReLU
        props['nonlin_kwargs'] = {'inplace': True}
    else:
        raise ValueError

    return props

class UNETR_v2(SegmentationNetwork): 

    def __init__(
        self,
            in_channels: int,
            out_channels: int,
            props,
            img_size: Union[Sequence[int], int],
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "conv",
            norm_name: Union[Tuple, str] = "in",
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            qkv_bias: bool = False,
            stages: Sequence[int] = [3,6,9,12],
            deep_supervision: bool = False, #after Ds
            upscale_logits=False,
            do_ds=False
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.feature_size = feature_size
        self.deep_supervision = deep_supervision #after Ds
        self.stages = stages
        self.num_layers = stages[-1]
        img_size = ensure_tuple_rep(img_size, spatial_dims) 
        self.patch_size = ensure_tuple_rep(16, spatial_dims) 
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        upsample_mode= "trilinear"
        self.do_ds = do_ds

        self.props = props #On ne veut pas de dropout
        self.deep_supervision_outputs = []
        self.conv_op = self.props['conv_op']
        self.num_classes = out_channels

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
        )

        self.encoders = nn.ModuleList()
        self.encoders.append(nn.Sequential(ConvDropoutNormReLU(input_channels=in_channels, output_channels= self.feature_size, kernel_size= np.array([3,3,3]), network_props= self.props), 
        (ConvDropoutNormReLU(input_channels=self.feature_size, output_channels= self.feature_size, kernel_size= np.array([3,3,3]), network_props= self.props))))
        """
        self.encoders = nn.ModuleList([StackedConvLayers(
            input_channels=in_channels,
            output_channels= self.feature_size,
            kernel_size=  np.array([3,3,3]),
            network_props= self.props,
            num_convs=2,
        )])"""

        for i in range(1,len(stages)):
            self.encoders.append(nn.Sequential(
            DeconvConvDropoutNormReLU(input_channels=hidden_size, output_channels=self.feature_size*(2**(i)), kernel_size=np.array([3,3,3]), network_props= self.props),
            *[DeconvConvDropoutNormReLU(input_channels=self.feature_size*(2**(i)), output_channels=self.feature_size*(2**(i)), kernel_size=np.array([3,3,3]), network_props= self.props) for _ in range(len(stages)-(i + 1))]
        ))
        
        self.decoders = nn.ModuleList()
        for  i in range(len(stages)-1):
            self.decoders.append(DecodeurLayer(
                input_channels=self.feature_size*2**(i+1),
                output_channels=self.feature_size*2**(i),
                kernel_size=np.array([3,3,3]),
                network_props=self.props
            ))

            if deep_supervision:
                seg_layer = self.props['conv_op'](in_channels=self.feature_size*2**(i), out_channels=out_channels, kernel_size=1, stride=1) #outchannels = 4
                self.deep_supervision_outputs.append(seg_layer)

        self.decoders.append(DecodeurLayer(
            input_channels=hidden_size,
            output_channels=self.feature_size*2**(len(stages)-1),
            kernel_size=np.array([3,3,3]),
            network_props=self.props
            ))

        if deep_supervision:    
            seg_layer = self.props['conv_op'](in_channels=self.feature_size*2**(len(stages)-1), out_channels=out_channels, kernel_size=1, stride=1) #outchannels = 4
            self.deep_supervision_outputs.append(seg_layer)
            self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

        self.out = self.props['conv_op'](in_channels=self.feature_size, out_channels=out_channels, kernel_size=1, stride=1)


        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]


        
    def proj_feat(self, x): 
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x
    

    
    
    def forward(self, x_in, gt=None, loss=None): 

        seg_outputs = []
        

        x, hidden_states_out = self.vit(x_in)
        enc = [self.encoders[0](x_in)]
        for i in range(1,len(self.encoders)):
            enc.append(self.encoders[i](self.proj_feat(hidden_states_out[self.stages[i-1]])))
        dec = self.proj_feat(x)
        for i in range(len(self.decoders)):
            dec = self.decoders[-i-1](dec, enc[-i-1])
 
            if self.deep_supervision and (i != len(self.decoders) - 1):
                tmp = self.deep_supervision_outputs[-i-1](dec)
                if gt is not None:
                    tmp = loss(tmp, gt)
                seg_outputs.append(tmp)
        
        dec = self.out(dec)
        if self.deep_supervision:
            tmp = dec
            if gt is not None:
                tmp = loss(tmp, gt)
            seg_outputs.append(tmp)
            return seg_outputs[::-1]
        else:
            return [dec]
        
    
