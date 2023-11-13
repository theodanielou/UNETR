from typing import Sequence, Union

import torch
import torch.nn as nn

import enum 

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

__all__ = ["ViT"]

from typing import Sequence, Type, Union, Any, Hashable, Mapping

from monai.utils import issequenceiterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from monai.networks.layers import Conv 
from monai.utils import ensure_tuple_rep, optional_import 
from monai.utils.module import look_up_option 

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}

import torch.nn as nn

from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock

from typing import Tuple, Union, Callable, Collection #OK 


from monai.networks.layers import get_act_layer 
from monai.utils import look_up_option #dans layers_factories.py
from monai.networks.layers.factories import Act, Dropout, Norm, Pool, split_args #dans layers_factories.py

SUPPORTED_DROPOUT_MODE = {"vit", "swin"}

import torch
import torch.nn as nn

from monai.utils import optional_import, min_version
Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")




class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            post_activation: add a final acivation function to the classification head when `classification` is True.
                Default to "Tanh" for `nn.Tanh()`. Set to other values to remove this function.
            qkv_bias: apply bias to the qkv linear layer in self attention block

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore


    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out

class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4, pos_embed="conv")

    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int,
        num_heads: int,
        pos_embed: str,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.pos_embed = look_up_option(pos_embed, SUPPORTED_EMBEDDING_TYPES)

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        self.patch_embeddings: nn.Module
        if self.pos_embed == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.pos_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: apply bias term for the qkv linear layer

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        act: Union[Tuple, str] = "GELU",
        dropout_mode="vit",
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: faction of the input units to drop.
            act: activation type and arguments. Defaults to GELU.
            dropout_mode: dropout mode, can be "vit" or "swin".
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = get_act_layer(act)
        self.drop1 = nn.Dropout(dropout_rate)
        dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
        if dropout_opt == "vit":
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_opt == "swin":
            self.drop2 = self.drop1
        else:
            raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")


    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: bias term for the qkv linear layer.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5


    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x

def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")

def optional_import(
    module: str,
    version: str = "",
    version_checker: Callable[..., bool] = min_version,
    name: str = "",
    descriptor: str = "{}", #OPTIONAL_IMPORT_MSG_FMT
    version_args=None,
    allow_namespace_pkg: bool = False,
    as_type: str = "default",
) -> Tuple[Any, bool]:
    """
    Imports an optional module specified by `module` string.
    Any importing related exceptions will be stored, and exceptions raise lazily
    when attempting to use the failed-to-import module.

    Args:
        module: name of the module to be imported.
        version: version string used by the version_checker.
        version_checker: a callable to check the module version, Defaults to monai.utils.min_version.
        name: a non-module attribute (such as method/class) to import from the imported module.
        descriptor: a format string for the final error message when using a not imported module.
        version_args: additional parameters to the version checker.
        allow_namespace_pkg: whether importing a namespace package is allowed. Defaults to False.
        as_type: there are cases where the optionally imported object is used as
            a base class, or a decorator, the exceptions should raise accordingly. The current supported values
            are "default" (call once to raise), "decorator" (call the constructor and the second call to raise),
            and anything else will return a lazy class that can be used as a base class (call the constructor to raise).

    Returns:
        The imported module and a boolean flag indicating whether the import is successful.

    Examples::

        >>> torch, flag = optional_import('torch', '1.1')
        >>> print(torch, flag)
        <module 'torch' from 'python/lib/python3.6/site-packages/torch/__init__.py'> True

        >>> the_module, flag = optional_import('unknown_module')
        >>> print(flag)
        False
        >>> the_module.method  # trying to access a module which is not imported
        OptionalImportError: import unknown_module (No module named 'unknown_module').

        >>> torch, flag = optional_import('torch', '42', exact_version)
        >>> torch.nn  # trying to access a module for which there isn't a proper version imported
        OptionalImportError: import torch (requires version '42' by 'exact_version').

        >>> conv, flag = optional_import('torch.nn.functional', '1.0', name='conv1d')
        >>> print(conv)
        <built-in method conv1d of type object at 0x11a49eac0>

        >>> conv, flag = optional_import('torch.nn.functional', '42', name='conv1d')
        >>> conv()  # trying to use a function from the not successfully imported module (due to unmatched version)
        OptionalImportError: from torch.nn.functional import conv1d (requires version '42' by 'min_version').
    """

    tb = None
    exception_str = ""
    if name:
        actual_cmd = f"from {module} import {name}"
    else:
        actual_cmd = f"import {module}"
    try:
        pkg = __import__(module)  # top level module
        the_module = import_module(module)
        if not allow_namespace_pkg:
            is_namespace = getattr(the_module, "__file__", None) is None and hasattr(the_module, "__path__")
            if is_namespace:
                raise AssertionError
        if name:  # user specified to load class/function/... from the module
            the_module = getattr(the_module, name)
    except Exception as import_exception:  # any exceptions during import
        tb = import_exception.__traceback__
        exception_str = f"{import_exception}"
    else:  # found the module
        if version_args and version_checker(pkg, f"{version}", version_args):
            return the_module, True
        if not version_args and version_checker(pkg, f"{version}"):
            return the_module, True

    # preparing lazy error message
    msg = descriptor.format(actual_cmd)
    if version and tb is None:  # a pure version issue
        msg += f" (requires '{module} {version}' by '{version_checker.__name__}')"
    if exception_str:
        msg += f" ({exception_str})"

    class _LazyRaise:
        def __init__(self, *_args, **_kwargs):
            _default_msg = (
                f"{msg}."
                + "\n\nFor details about installing the optional dependencies, please visit:"
                + "\n    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies"
            )
            if tb is None:
                self._exception = OptionalImportError(_default_msg)
            else:
                self._exception = OptionalImportError(_default_msg).with_traceback(tb)

        def __getattr__(self, name):
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

        def __call__(self, *_args, **_kwargs):
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

        def __getitem__(self, item):
            raise self._exception

        def __iter__(self):
            raise self._exception

    if as_type == "default":
        return _LazyRaise(), False

    class _LazyCls(_LazyRaise):
        def __init__(self, *_args, **kwargs):
            super().__init__()
            if not as_type.startswith("decorator"):
                raise self._exception

    return _LazyCls, False


def look_up_option(opt_str, supported: Union[Collection, enum.EnumMeta], default="no_default", print_all_options=True):
    """
    Look up the option in the supported collection and return the matched item.
    Raise a value error possibly with a guess of the closest match.

    Args:
        opt_str: The option string or Enum to look up.
        supported: The collection of supported options, it can be list, tuple, set, dict, or Enum.
        default: If it is given, this method will return `default` when `opt_str` is not found,
            instead of raising a `ValueError`. Otherwise, it defaults to `"no_default"`,
            so that the method may raise a `ValueError`.
        print_all_options: whether to print all available options when `opt_str` is not found. Defaults to True

    Examples:

    .. code-block:: python

        from enum import Enum
        from monai.utils import look_up_option
        class Color(Enum):
            RED = "red"
            BLUE = "blue"
        look_up_option("red", Color)  # <Color.RED: 'red'>
        look_up_option(Color.RED, Color)  # <Color.RED: 'red'>
        look_up_option("read", Color)
        # ValueError: By 'read', did you mean 'red'?
        # 'read' is not a valid option.
        # Available options are {'blue', 'red'}.
        look_up_option("red", {"red", "blue"})  # "red"

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/utilities/util_common.py#L249
    """
    if not isinstance(opt_str, Hashable):
        raise ValueError(f"Unrecognized option type: {type(opt_str)}:{opt_str}.")
    if isinstance(opt_str, str):
        opt_str = opt_str.strip()
    if isinstance(supported, enum.EnumMeta):
        if isinstance(opt_str, str) and opt_str in {item.value for item in cast(Iterable[enum.Enum], supported)}:
            # such as: "example" in MyEnum
            return supported(opt_str)
        if isinstance(opt_str, enum.Enum) and opt_str in supported:
            # such as: MyEnum.EXAMPLE in MyEnum
            return opt_str
    elif isinstance(supported, Mapping) and opt_str in supported:
        # such as: MyDict[key]
        return supported[opt_str]
    elif isinstance(supported, Collection) and opt_str in supported:
        return opt_str

    if default != "no_default":
        return default

    # find a close match
    set_to_check: set
    if isinstance(supported, enum.EnumMeta):
        set_to_check = {item.value for item in cast(Iterable[enum.Enum], supported)}
    else:
        set_to_check = set(supported) if supported is not None else set()
    if not set_to_check:
        raise ValueError(f"No options available: {supported}.")
    edit_dists = {}
    opt_str = f"{opt_str}"
    for key in set_to_check:
        edit_dist = damerau_levenshtein_distance(f"{key}", opt_str)
        if edit_dist <= 3:
            edit_dists[key] = edit_dist

    supported_msg = f"Available options are {set_to_check}.\n" if print_all_options else ""
    if edit_dists:
        guess_at_spelling = min(edit_dists, key=edit_dists.get)  # type: ignore
        raise ValueError(
            f"By '{opt_str}', did you mean '{guess_at_spelling}'?\n"
            + f"'{opt_str}' is not a valid value.\n"
            + supported_msg
        )
    raise ValueError(f"Unsupported option '{opt_str}', " + supported_msg)


def get_act_layer(name: Union[Tuple, str]):
    """
    Create an activation layer instance.

    For example, to create activation layers:

    .. code-block:: python

        from monai.networks.layers import get_act_layer

        s_layer = get_act_layer(name="swish")
        p_layer = get_act_layer(name=("prelu", {"num_parameters": 1, "init": 0.25}))

    Args:
        name: an activation type string or a tuple of type string and parameters.
    """
    if name == "":
        return torch.nn.Identity()
    act_name, act_args = split_args(name)
    act_type = Act[act_name]
    return act_type(**act_args)
