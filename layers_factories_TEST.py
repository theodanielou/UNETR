# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import warnings
from typing import Any, Callable, Dict, Tuple, Type, Union

import torch
import torch.nn as nn

from monai.utils import look_up_option, optional_import

""" Pour look_up_option """
import enum
from typing import Collection, Hashable, Iterable, Mapping, Union, cast

""" Pour optional_import"""
from importlib import import_module
from typing import Any, Callable, Tuple


InstanceNorm3dNVFuser, has_nvfuser = optional_import("apex.normalization", name="InstanceNorm3dNVFuser")

__all__ = ["LayerFactory", "Dropout", "Norm", "Act", "Conv", "Pool", "Pad", "split_args"]


class LayerFactory:
    """
    Factory object for creating layers, this uses given factory functions to actually produce the types or constructing
    callables. These functions are referred to by name and can be added at any time.
    """

    def __init__(self) -> None:
        self.factories: Dict[str, Callable] = {}

    @property
    def names(self) -> Tuple[str, ...]:
        """
        Produces all factory names.
        """

        return tuple(self.factories)

    def add_factory_callable(self, name: str, func: Callable) -> None:
        """
        Add the factory function to this object under the given name.
        """

        self.factories[name.upper()] = func
        self.__doc__ = (
            "The supported member"
            + ("s are: " if len(self.names) > 1 else " is: ")
            + ", ".join(f"``{name}``" for name in self.names)
            + ".\nPlease see :py:class:`monai.networks.layers.split_args` for additional args parsing."
        )


    def factory_function(self, name: str) -> Callable:
        """
        Decorator for adding a factory function with the given name.
        """

        def _add(func: Callable) -> Callable:
            self.add_factory_callable(name, func)
            return func

        return _add


    def get_constructor(self, factory_name: str, *args) -> Any:
        """
        Get the constructor for the given factory name and arguments.

        Raises:
            TypeError: When ``factory_name`` is not a ``str``.

        """

        if not isinstance(factory_name, str):
            raise TypeError(f"factory_name must a str but is {type(factory_name).__name__}.")

        func = look_up_option(factory_name.upper(), self.factories)
        return func(*args)


    def __getitem__(self, args) -> Any:
        """
        Get the given name or name/arguments pair. If `args` is a callable it is assumed to be the constructor
        itself and is returned, otherwise it should be the factory name or a pair containing the name and arguments.
        """

        # `args[0]` is actually a type or constructor
        if callable(args):
            return args

        # `args` is a factory name or a name with arguments
        if isinstance(args, str):
            name_obj, args = args, ()
        else:
            name_obj, *args = args

        return self.get_constructor(name_obj, *args)

    def __getattr__(self, key):
        """
        If `key` is a factory name, return it, otherwise behave as inherited. This allows referring to factory names
        as if they were constants, eg. `Fact.FOO` for a factory Fact with factory function foo.
        """

        if key in self.factories:
            return key

        return super().__getattribute__(key)


def split_args(args):
    """
    Split arguments in a way to be suitable for using with the factory types. If `args` is a string it's interpreted as
    the type name.

    Args:
        args (str or a tuple of object name and kwarg dict): input arguments to be parsed.

    Raises:
        TypeError: When ``args`` type is not in ``Union[str, Tuple[Union[str, Callable], dict]]``.

    Examples::

        >>> act_type, args = split_args("PRELU")
        >>> monai.networks.layers.Act[act_type]
        <class 'torch.nn.modules.activation.PReLU'>

        >>> act_type, args = split_args(("PRELU", {"num_parameters": 1, "init": 0.25}))
        >>> monai.networks.layers.Act[act_type](**args)
        PReLU(num_parameters=1)

    """

    if isinstance(args, str):
        return args, {}
    name_obj, name_args = args

    if not (isinstance(name_obj, str) or callable(name_obj)) or not isinstance(name_args, dict):
        msg = "Layer specifiers must be single strings or pairs of the form (name/object-types, argument dict)"
        raise TypeError(msg)

    return name_obj, name_args



# Define factories for these layer types

Dropout = LayerFactory()
Norm = LayerFactory()
Act = LayerFactory()
Conv = LayerFactory()
Pool = LayerFactory()
Pad = LayerFactory()


@Dropout.factory_function("dropout")
def dropout_factory(dim: int) -> Type[Union[nn.Dropout, nn.Dropout2d, nn.Dropout3d]]:
    types = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
    return types[dim - 1]


@Dropout.factory_function("alphadropout")
def alpha_dropout_factory(_dim):
    return nn.AlphaDropout


@Norm.factory_function("instance")
def instance_factory(dim: int) -> Type[Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]]:
    types = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    return types[dim - 1]


@Norm.factory_function("batch")
def batch_factory(dim: int) -> Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]]:
    types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    return types[dim - 1]


@Norm.factory_function("group")
def group_factory(_dim) -> Type[nn.GroupNorm]:
    return nn.GroupNorm


@Norm.factory_function("layer")
def layer_factory(_dim) -> Type[nn.LayerNorm]:
    return nn.LayerNorm


@Norm.factory_function("localresponse")
def local_response_factory(_dim) -> Type[nn.LocalResponseNorm]:
    return nn.LocalResponseNorm


@Norm.factory_function("syncbatch")
def sync_batch_factory(_dim) -> Type[nn.SyncBatchNorm]:
    return nn.SyncBatchNorm


@Norm.factory_function("instance_nvfuser")
def instance_nvfuser_factory(dim):
    """
    `InstanceNorm3dNVFuser` is a faster version of InstanceNorm layer and implemented in `apex`.
    It only supports 3d tensors as the input. It also requires to use with CUDA and non-Windows OS.
    In this function, if the required library `apex.normalization.InstanceNorm3dNVFuser` does not exist,
    `nn.InstanceNorm3d` will be returned instead.
    This layer is based on a customized autograd function, which is not supported in TorchScript currently.
    Please switch to use `nn.InstanceNorm3d` if TorchScript is necessary.

    Please check the following link for more details about how to install `apex`:
    https://github.com/NVIDIA/apex#installation

    """
    types = (nn.InstanceNorm1d, nn.InstanceNorm2d)
    if dim != 3:
        warnings.warn(f"`InstanceNorm3dNVFuser` only supports 3d cases, use {types[dim - 1]} instead.")
        return types[dim - 1]
    # test InstanceNorm3dNVFuser installation with a basic example
    has_nvfuser_flag = has_nvfuser
    if not torch.cuda.is_available():
        return nn.InstanceNorm3d
    try:
        layer = InstanceNorm3dNVFuser(num_features=1, affine=True).to("cuda:0")
        inp = torch.randn([1, 1, 1, 1, 1]).to("cuda:0")
        out = layer(inp)
        del inp, out, layer
    except Exception:
        has_nvfuser_flag = False
    if not has_nvfuser_flag:
        warnings.warn(
            "`apex.normalization.InstanceNorm3dNVFuser` is not installed properly, use nn.InstanceNorm3d instead."
        )
        return nn.InstanceNorm3d
    return InstanceNorm3dNVFuser


Act.add_factory_callable("elu", lambda: nn.modules.ELU)
Act.add_factory_callable("relu", lambda: nn.modules.ReLU)
Act.add_factory_callable("leakyrelu", lambda: nn.modules.LeakyReLU)
Act.add_factory_callable("prelu", lambda: nn.modules.PReLU)
Act.add_factory_callable("relu6", lambda: nn.modules.ReLU6)
Act.add_factory_callable("selu", lambda: nn.modules.SELU)
Act.add_factory_callable("celu", lambda: nn.modules.CELU)
Act.add_factory_callable("gelu", lambda: nn.modules.GELU)
Act.add_factory_callable("sigmoid", lambda: nn.modules.Sigmoid)
Act.add_factory_callable("tanh", lambda: nn.modules.Tanh)
Act.add_factory_callable("softmax", lambda: nn.modules.Softmax)
Act.add_factory_callable("logsoftmax", lambda: nn.modules.LogSoftmax)


@Act.factory_function("swish")
def swish_factory():
    from monai.networks.blocks.activation import Swish

    return Swish


@Act.factory_function("memswish")
def memswish_factory():
    from monai.networks.blocks.activation import MemoryEfficientSwish

    return MemoryEfficientSwish


@Act.factory_function("mish")
def mish_factory():
    from monai.networks.blocks.activation import Mish

    return Mish


@Conv.factory_function("conv")
def conv_factory(dim: int) -> Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]]:
    types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    return types[dim - 1]


@Conv.factory_function("contvrans")
def convtrans_factory(dim: int) -> Type[Union[nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]]:
    types = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    return types[dim - 1]


@Pool.factory_function("max")
def maxpooling_factory(dim: int) -> Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]]:
    types = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)
    return types[dim - 1]


@Pool.factory_function("adaptivemax")
def adaptive_maxpooling_factory(
    dim: int,
) -> Type[Union[nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d]]:
    types = (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)
    return types[dim - 1]


@Pool.factory_function("avg")
def avgpooling_factory(dim: int) -> Type[Union[nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]]:
    types = (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)
    return types[dim - 1]


@Pool.factory_function("adaptiveavg")
def adaptive_avgpooling_factory(
    dim: int,
) -> Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]]:
    types = (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)
    return types[dim - 1]


@Pad.factory_function("replicationpad")
def replication_pad_factory(dim: int) -> Type[Union[nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d]]:
    types = (nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d)
    return types[dim - 1]


@Pad.factory_function("constantpad")
def constant_pad_factory(dim: int) -> Type[Union[nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d]]:
    types = (nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d)
    return types[dim - 1]


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


def optional_import(
    module: str,
    version: str = "",
    version_checker: Callable[..., bool] = min_version,
    name: str = "",
    descriptor: str = OPTIONAL_IMPORT_MSG_FMT,
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