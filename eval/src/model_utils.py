import re
import os
from types import MethodType
from collections import defaultdict
from typing import List, Dict, Optional, Union, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from src.common_utils import to


### Layer and activation getters


class CatcherExit(Exception):
    pass


class Catcher(nn.Module):

    def __init__(self, module: nn.Module, offload: bool = False):
        super().__init__()
        self.module = module
        self.inputs = []
        self.input_kwargs = []
        self.offload = offload

    def forward(self, inputs, **kwargs):
        offload_device = "cpu" if self.offload else None
        self.inputs.append(inputs.to(offload_device))
        self.input_kwargs.append(kwargs)
        raise CatcherExit()


def get_layers(model: AutoModelForCausalLM):
    if model.config.model_type in ("llama", "gemma", "gemma2", "phi3", "mistral"):
        return model.model.layers
    if model.config.model_type == "opt":
        return model.model.decoder.layers
    else:
        raise ValueError(f"{model.config.model_type} is not supported.")


def get_lm_head(model: AutoModelForCausalLM):
    lm_head = nn.ModuleList()
    if model.config.model_type in ("llama", "gemma", "gemma2", "phi3", "mistral"):
        if model.model.norm is not None:
            lm_head.append(model.model.norm)
        lm_head.append(model.lm_head)
    elif model.config.model_type == "opt":
        if model.model.decoder.final_layer_norm is not None:
            lm_head.append(model.model.decoder.final_layer_norm)
        if model.model.decoder.project_out is not None:
            lm_head.append(model.model.decoder.project_out)
        lm_head.append(model.lm_head)
    else:
        raise ValueError(f"{model.config.model_type} is not supported.")
    return lm_head


def get_transformer_block_class(model: AutoModelForCausalLM):
    if model.config.model_type == "llama":
        return LlamaDecoderLayer
    if model.config.model_type == "opt":
        return OPTDecoderLayer
    if model.config.model_type == "gemma2":
        return Gemma2DecoderLayer
    if model.config.model_type == "phi3":
        return Phi3DecoderLayer
    if model.config.model_type == "mistral":
        return MistralDecoderLayer
    else:
        raise ValueError(f"{model.config.model_type} is not supported.")


def get_mlp_layer_name(model: AutoModelForCausalLM):
    if model.config.model_type in ("llama", "mistral"):
        return "mlp"
    else:
        raise ValueError(f"{model.config.model_type} is not supported.")


def get_attn_layer_name(model: AutoModelForCausalLM):
    if model.config.model_type in ("llama", "mistral"):
        return "self_attn"
    
    else:
        raise ValueError(f"{model.config.model_type} is not supported.")


def get_lm_logits(hidden_states: torch.Tensor, model: nn.Module):
    if model.config.model_type in ("llama", "gemma", "gemma2", "phi3", "mistral"):
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == "opt":
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    else:
        raise ValueError(f"{model.config.model_type} is not supported.")
    return lm_logits


def get_shifted_lm_logits(hidden_states: torch.Tensor, model: nn.Module, flatten: bool = False):
    shifted_lm_logits = get_lm_logits(hidden_states, model)[:, :-1, :].contiguous()
    if flatten:
        shifted_lm_logits = shifted_lm_logits.flatten(0, -2)
    return shifted_lm_logits


def get_hidden_size(model: AutoModelForCausalLM):
    if model.config.model_type in ("llama", "gemma", "gemma2", "phi3", "opt", "mistral"):
        return model.config.hidden_size
    else:
        raise ValueError(f"{model.config.model_type} is not supported.")


### Zero/Identity module utilities
def dummy_initialize(module: nn.Module) -> None:
    module.__forward = module.forward


def make_dummy_forward(module: nn.Module, layer_type: str = "attn+mlp") -> None:
    assert layer_type in ["attn+mlp", "attn", "mlp"]
    # Forward that returns first argument
    if layer_type == "attn+mlp":

        def dummy_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
            return (hidden_states,)

    elif layer_type == "attn":

        def dummy_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
            return 0, None, None

    elif layer_type == "mlp":

        def dummy_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
            return 0

    module.forward = MethodType(dummy_forward, module)


def restore_forward(module: nn.Module) -> None:
    module.forward = module.__forward


class ZeroMLP(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        return 0


class ZeroAttention(nn.Module):

    def __init__(self, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor, past_key_value=None, *args, **kwargs):
        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None and hasattr(past_key_value, "update"):
            past_key_value.update(torch.empty(1, 1, 1), torch.empty(1, 1, 1), self.layer_idx, {})
        return 0, None, past_key_value


class IdentityLayer(nn.Module):

    def __init__(self, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        *args,
        **kwargs,
    ):
        outputs = (hidden_states,)

        if past_key_value is not None and hasattr(past_key_value, "update"):
            past_key_value.update(torch.empty(1, 1, 1), torch.empty(1, 1, 1), self.layer_idx, {})

        if output_attentions is not None:
            outputs += (None,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


def drop_layers(model, drop_config: List[int]):
    layers = get_layers(model)
    attn_layer_name = get_attn_layer_name(model)
    mlp_layer_name = get_mlp_layer_name(model)

    assert len(layers) == len(drop_config)

    for layer_id, _ in enumerate(layers):
        # Do nothing
        if drop_config[layer_id] == "none":
            pass
        # Remove mlp
        elif drop_config[layer_id] == "mlp":
            setattr(layers[layer_id], mlp_layer_name, ZeroMLP())
        # Remove attention
        elif drop_config[layer_id] == "attn":
            setattr(layers[layer_id], attn_layer_name, ZeroAttention(layer_idx=layer_id))
        # Remove both mlp and attention
        elif drop_config[layer_id] == "attn+mlp":
            layers[layer_id] = IdentityLayer(layer_idx=layer_id)


def drop_layers_from_config(model, drop_config_path: str):
    drop_config = []
    with open(drop_config_path, "r") as f:
        for line in f:
            drop_config.append(line.strip("\n"))
    drop_layers(model, drop_config)


### Calibration utils and modules

LINEAR_LAYERS = (nn.Linear, _ConvNd)


class ForwardInterrupt(Exception):
    pass


class InputCollector(nn.Module):

    def __init__(self, module: nn.Module, cpu_offload: bool = False):
        super().__init__()
        self.module = module
        self.cpu_offload = cpu_offload
        self.input_args = []
        self.input_kwargs = []

    def forward(self, *input_args, **input_kwargs):
        """
        Assumes that the wrapped module has a single
        input that can reside in inputs or input_kwargs.
        """
        if self.cpu_offload:
            input_args = to(input_args, device="cpu")
            input_kwargs = to(input_kwargs, device="cpu")
        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)
        raise ForwardInterrupt


def select_layers(
    model: nn.Module,
    layer_prefix: Optional[str] = "",
    layer_regex: str = ".*",
    layer_classes: Union[nn.Module, List[nn.Module]] = nn.Module,
) -> Dict[str, nn.Module]:
    layers = {}
    for layer_name, layer in model.named_modules():
        if (
            isinstance(layer, layer_classes)
            and re.search(layer_regex, layer_name)
            and layer_name.startswith(layer_prefix)
        ):
            layers[layer_name] = layer
    return layers


def get_number_of_rows_and_cols(layer):
    return layer.weight.shape[0], np.prod(layer.weight.shape[1:])


### Feature extraction utils


class FeatureExtractorWrapper(nn.Module):

    def __init__(self, model: nn.Module, module_regex: str):
        super().__init__()
        self.model = model
        self.cache_features = False  # if True - cache features
        self.forward_hooks = {}
        self.cached_features = {}
        for module_name, module in self.model.named_modules():
            # Remove _fsdp parts from module name
            module_name = ".".join([x for x in module_name.split(".") if x != "_fsdp_wrapped_module"])
            if re.search(module_regex, module_name):

                def cache_output(mod_name):
                    def hook(mod, inputs, outputs):
                        if self.cache_features:
                            if isinstance(outputs, Sequence):
                                outputs = outputs[0]
                            self.cached_features[mod_name] = outputs

                    return hook

                self.forward_hooks[module_name] = module.register_forward_hook(cache_output(module_name))

    def clean_cache(self):
        self.cached_features = {}

    def clean_all(self):
        for _, hook in self.forward_hooks():
            hook.remove()
        self.cached_features = {}

    def forward(self, *input_args, **input_kwargs):
        output = self.model(*input_args, **input_kwargs)
        output.features = self.cached_features
        return output


### Sparse model loader


def load_sparse_weights(
    model: AutoModelForCausalLM,
    sparse_weights_path: Union[str, os.PathLike],
    sparse_config_path: Optional[str] = None,
    default_level: int = 0,
):
    # Load weights from configuration if provided
    if sparse_config_path:
        with open(os.path.join(sparse_weights_path, sparse_config_path), "r") as f:
            for line in f:
                layer_name, level = line.split(":")
                layer = model.get_submodule(layer_name.strip(" "))
                orig_dtype = layer.weight.dtype
                layer.weight.data = torch.load(
                    os.path.join(sparse_weights_path, layer_name, f"{int(level)}.pth"),
                ).to(orig_dtype)
    # Otherwise load uniform configuration
    else:
        for layer_name in sorted(os.listdir(sparse_weights_path)):
            if not os.path.isdir(os.path.join(sparse_weights_path, layer_name)):
                continue
            layer = model.get_submodule(layer_name.strip(" "))
            orig_dtype = layer.weight.dtype
            layer.weight.data = torch.load(
                os.path.join(sparse_weights_path, layer_name, f"{default_level}.pth"),
            ).to(orig_dtype)


def layer_order_fn(layer_name: str):
    split_key = layer_name.split(".")
    block_id = int(split_key[2])
    misc = split_key[3:]
    return (block_id, *misc)

def group_layers(model: nn.Module, layer_names: Sequence[str], group_rule: Optional[str] = None) -> Tuple[Sequence[str]]:
    assert group_rule in ["none", "name", "size"]
    # No grouping
    if group_rule == "none":
        group_key_fn = lambda layer_name: 0
    # Group by last part of the name
    elif group_rule == "name":
        group_key_fn = lambda layer_name: layer_name.split(".")[-1]
    # Group by size
    elif group_rule == "size":
        group_key_fn = lambda layer_name: model.get_submodule(layer_name).weight.numel()
    groups = defaultdict(list)
    for layer_name in layer_names:
        groups[group_key_fn(layer_name)].append(layer_name)
    return tuple(groups.values())