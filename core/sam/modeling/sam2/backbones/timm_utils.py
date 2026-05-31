from .repvit import RepVit
from copy import deepcopy
from typing import List, Tuple, Union, Sequence, Dict, Optional
from torch import nn
import torch
from collections import OrderedDict
import os

OutIndicesT = Union[int, Tuple[int, ...]]

if 'TIMM_REENTRANT_CKPT' in os.environ:
    _USE_REENTRANT_CKPT = bool(os.environ['TIMM_REENTRANT_CKPT'])
else:
    _USE_REENTRANT_CKPT = False

def use_reentrant_ckpt() -> bool:
    return _USE_REENTRANT_CKPT


def checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    **kwargs,
):

    if use_reentrant is None:
        use_reentrant = use_reentrant_ckpt()

    return torch.utils.checkpoint.checkpoint(
        function,
        *args,
        use_reentrant=use_reentrant,
        **kwargs,
    )

def _out_indices_as_tuple(x: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return tuple(range(-x, 0))
    return tuple(x)

class FeatureInfo:

    def __init__(
            self,
            feature_info: List[Dict],
            out_indices: OutIndicesT,
    ):
        out_indices = _out_indices_as_tuple(out_indices)
        prev_reduction = 1
        for i, fi in enumerate(feature_info):
            # sanity check the mandatory fields, there may be additional fields depending on the model
            assert 'num_chs' in fi and fi['num_chs'] > 0
            assert 'reduction' in fi and fi['reduction'] >= prev_reduction
            prev_reduction = fi['reduction']
            assert 'module' in fi
            fi.setdefault('index', i)
        self.out_indices = out_indices
        self.info = feature_info

    def from_other(self, out_indices: OutIndicesT):
        out_indices = _out_indices_as_tuple(out_indices)
        return FeatureInfo(deepcopy(self.info), out_indices)

    def get(self, key: str, idx: Optional[Union[int, List[int]]] = None):

        if idx is None:
            return [self.info[i][key] for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i][key] for i in idx]
        else:
            return self.info[idx][key]

    def get_dicts(self, keys: Optional[List[str]] = None, idx: Optional[Union[int, List[int]]] = None):

        if idx is None:
            if keys is None:
                return [self.info[i] for i in self.out_indices]
            else:
                return [{k: self.info[i][k] for k in keys} for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i] if keys is None else {k: self.info[i][k] for k in keys} for i in idx]
        else:
            return self.info[idx] if keys is None else {k: self.info[idx][k] for k in keys}

    def channels(self, idx: Optional[Union[int, List[int]]] = None):

        return self.get('num_chs', idx)

    def reduction(self, idx: Optional[Union[int, List[int]]] = None):

        return self.get('reduction', idx)

    def module_name(self, idx: Optional[Union[int, List[int]]] = None):


        return self.get('module', idx)

    def __getitem__(self, item):
        return self.info[item]

    def __len__(self):
        return len(self.info)

def _get_feature_info(net, out_indices: OutIndicesT):
    feature_info = getattr(net, 'feature_info')
    if isinstance(feature_info, FeatureInfo):
        return feature_info.from_other(out_indices)
    elif isinstance(feature_info, (list, tuple)):
        return FeatureInfo(net.feature_info, out_indices)
    else:
        assert False, "Provided feature_info is not valid"

def _get_return_layers(feature_info, out_map):
    module_names = feature_info.module_name()
    return_layers = {}
    for i, name in enumerate(module_names):
        return_layers[name] = out_map[i] if out_map is not None else feature_info.out_indices[i]
    return return_layers

def _module_list(module, flatten_sequential=False):
    ml = []
    for name, module in module.named_children():
        if flatten_sequential and isinstance(module, nn.Sequential):
            for child_name, child_module in module.named_children():
                combined = [name, child_name]
                ml.append(('_'.join(combined), '.'.join(combined), child_module))
        else:
            ml.append((name, name, module))
    return ml

class FeatureDictNet(nn.ModuleDict):

    def __init__(
            self,
            model: nn.Module,
            out_indices: OutIndicesT = (0, 1, 2, 3, 4),
            out_map: Sequence[Union[int, str]] = None,
            output_fmt: str = 'NCHW',
            feature_concat: bool = False,
            flatten_sequential: bool = False,
    ):

        super(FeatureDictNet, self).__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        self.output_fmt = output_fmt
        self.concat = feature_concat
        self.grad_checkpointing = False
        self.return_layers = {}

        return_layers = _get_return_layers(self.feature_info, out_map)
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = set(return_layers.keys())
        layers = OrderedDict()
        for new_name, old_name, module in modules:
            layers[new_name] = module
            if old_name in remaining:
                # return id has to be consistently str type for torchscript
                self.return_layers[new_name] = str(return_layers[old_name])
                remaining.remove(old_name)
            if not remaining:
                break
        assert not remaining and len(self.return_layers) == len(return_layers), \
            f'Return layers ({remaining}) are not present in model'
        self.update(layers)

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def _collect(self, x) -> (Dict[str, torch.Tensor]):
        out = OrderedDict()
        for i, (name, module) in enumerate(self.items()):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                first_or_last_module = i == 0 or i == max(len(self) - 1, 0)
                x = module(x) if first_or_last_module else checkpoint(module, x)
            else:
                x = module(x)

            if name in self.return_layers:
                out_id = self.return_layers[name]
                if isinstance(x, (tuple, list)):
                    out[out_id] = torch.cat(x, 1) if self.concat else x[0]
                else:
                    out[out_id] = x
        return out

    def forward(self, x) -> Dict[str, torch.Tensor]:
        return self._collect(x)

class FeatureListNet(FeatureDictNet):

    def __init__(
            self,
            model: nn.Module,
            out_indices: OutIndicesT = (0, 1, 2, 3, 4),
            output_fmt: str = 'NCHW',
            feature_concat: bool = False,
            flatten_sequential: bool = False,
    ):
        super().__init__(
            model,
            out_indices=out_indices,
            output_fmt=output_fmt,
            feature_concat=feature_concat,
            flatten_sequential=flatten_sequential,
        )

    def forward(self, x) -> (List[torch.Tensor]):
        return list(self._collect(x).values())

def build_repvit_with_cfg(
    variant: str,
    features_only: bool = False,
    out_indices=(0, 1, 2, 3),
    **kwargs,
):

    if variant.startswith("repvit_m1"):
        embed_dim = (48, 96, 192, 384)
        depth = (2, 2, 14, 2)
    else:
        raise NotImplementedError(f"Unknown RepVit variant: {variant}")

    model = RepVit(
        embed_dim=embed_dim,
        depth=depth,
        in_chans=3,
        num_classes=1000,
        **kwargs,
    )


    if features_only:
        model = FeatureListNet(model, out_indices=out_indices, flatten_sequential=True)

    return model

# 사용 예시
if __name__ == "__main__":
    model = build_repvit_with_cfg(
        "repvit_m1.dist_in1k",
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )

