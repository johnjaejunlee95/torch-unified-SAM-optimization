import sys
import functools
import inspect
from collections.abc import Mapping
from types import ModuleType

from .ESAM import ESAM
from .SAM import *
from .FisherSAM import *
from .LookSAM import *
from .GSAM import *
from .bSAM import *
from .FriendlySAM import FriendlySAM

_SAM_VARIANT_CLASSES = {
    "SAM": SAM,
    "ASAM": SAM,
    "ESAM": ESAM,
    "FisherSAM": FisherSAM,
    "LookSAM": LookSAM,
    "GSAM": GSAM,
    "BayesianSAM": BayesianSAM,
    "FriendlySAM": FriendlySAM,
}

COMMON_OPTIMIZER_PARAMS = {
    'lr', 'weight_decay', 'momentum', 'dampening',
    'betas', 'eps', 'maximize', 'foreach', 
    'capturable', 'differentiable', 'fused', 'amsgrad'
}


def _filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls)
    params = sig.parameters

    if all(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    
    allowed = {
        name
        for name, param in params.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    
    allowed.update(COMMON_OPTIMIZER_PARAMS)
    
    return {k: v for k, v in kwargs.items() if k in allowed}


class SAMVariantRegistry(Mapping):
    def __init__(self, variants):
        self._variants = dict(variants)
        self._factories = {}

    def __getitem__(self, name):
        if name not in self._factories:
            cls = self._variants[name]

            @functools.wraps(cls)
            def factory(*args, **kwargs):
                filtered_kwargs = _filter_kwargs(cls, kwargs)
                
                # --- MODIFICATION START ---
                # Explicitly remove weight_decay for BayesianSAM
                if name == "BayesianSAM":
                    filtered_kwargs.pop('weight_decay', None)
                    filtered_kwargs.pop('momentum', None)
                # --- MODIFICATION END ---
                
                return cls(*args, **filtered_kwargs)

            factory.cls = cls
            self._factories[name] = factory
        return self._factories[name]

    def __iter__(self):
        return iter(self._variants)

    def __len__(self):
        return len(self._variants)

    def raw(self, name):
        return self._variants[name]


SAM_Varients = SAMVariantRegistry(_SAM_VARIANT_CLASSES)


def build_sam_variant(name, *args, **kwargs):
    return SAM_Varients[name](*args, **kwargs)


def __getattr__(name):
    try:
        return _SAM_VARIANT_CLASSES[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc


class _SAMVariantModule(ModuleType):
    def __getitem__(self, name):
        return SAM_Varients[name]


sys.modules[__name__].__class__ = _SAMVariantModule