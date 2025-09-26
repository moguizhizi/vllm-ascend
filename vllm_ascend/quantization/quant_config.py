#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod, VocabParallelEmbedding)
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.distributed.parallel_state import (get_mlp_tp_group,
                                                    get_otp_group)
from vllm_ascend.ops.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.utils import (ASCEND_QUANTIZATION_METHOD, mlp_tp_enable,
                               oproj_tp_enable)

from .utils import get_quant_method

from collections.abc import Iterable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


@register_quantization_config(ASCEND_QUANTIZATION_METHOD)
class AscendQuantConfig(QuantizationConfig):
    """Config class for Ascend

    This class is a general class that parse quantization configs
    that are supported on ascend hardware.
    """

    def __init__(self, quant_config: Dict[str, Any]):
        super().__init__()
        self.quant_description = quant_config
        self.ignore_prefixes, self.all_prefixes = self.get_excluded_layer_prefixes(quant_config)
        self.hf_to_vllm_name_map: dict[str, str] = dict() 

    def __repr__(self) -> str:
        return "AscendQuantConfig:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return ASCEND_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_model_description.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AscendQuantConfig":
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        if torch.npu.is_available():
            return ASCEND_QUANTIZATION_METHOD
        return None
    
    def get_excluded_layer_prefixes(self, quant_config: Dict[str, Any]) -> tuple[list[str], list[str]]:
        """获取需要忽略的层和所有层的前缀列表"""
        ignore_prefixes = set()
        all_prefixes = set()

        for key, value in quant_config.items():
            if key == "model_quant_type":
                continue

            # 获取层前缀（去掉最后一个字段）
            prefix = key.rsplit(".", 1)[0] if "." in key else key

            # 只有当 key 以 "weight" 结尾 且 value == "FLOAT" 时才忽略
            if key.endswith("weight") and value == "FLOAT":
                ignore_prefixes.add(prefix)

            all_prefixes.add(prefix)

        return list(ignore_prefixes), list(all_prefixes)
    
    def apply_list(self, values: list[str], hf_to_vllm_mapper: "WeightsMapper" ) -> dict[str, str]:


        return {
            out_name: name
            for name in values
            if (out_name := hf_to_vllm_mapper._map_name(name)) is not None
        }

    def apply_fields(self, hf_to_vllm_mapper: "WeightsMapper",
                     field_map: dict[str, list[str]],
                     as_keys: set[str] | None = None,):
        """
        通用映射工具：
        - field_map: {attr_name: values_to_map}
        - as_keys: 哪些属性只保留 keys，而不是完整的 dict
        """

        as_keys = as_keys or set()
        for attr, values in field_map.items():
            mapped = self.apply_list(values, hf_to_vllm_mapper)
            if attr in as_keys:
                setattr(self, attr, list(mapped.keys()))
            else:
                setattr(self, attr, mapped)

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        self.apply_fields(
            hf_to_vllm_mapper,
            field_map={
                "hf_to_vllm_name_map": self.all_prefixes,
                "ignore_prefixes": self.ignore_prefixes,
            },
            as_keys={"ignore_prefixes"},
        )
    
    def get_origin_prefix(self, prefix: str) -> str:
        packed_mapping = getattr(self, "packed_modules_mapping", None)
        if not isinstance(packed_mapping, dict) or not packed_mapping:
            return prefix

        proj_name = prefix.split(".")[-1]
        shard_proj_names = packed_mapping.get(proj_name)
        if not shard_proj_names:
            return self.hf_to_vllm_name_map.get(prefix, prefix)

        for shard_proj_name in shard_proj_names:
            shard_name = prefix.rsplit(".", 1)[0] + "." + shard_proj_name
            mapped_name = self.hf_to_vllm_name_map.get(shard_name)
            if mapped_name:
                # 替换最后一段 shard_proj_name -> proj_name
                return mapped_name.rsplit(".", 1)[0] + "." + proj_name

        return prefix
    
    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention

        origin_prefix = self.get_origin_prefix(prefix)

        if isinstance(layer, LinearBase):
            if self.should_ignore_layer(prefix, self.ignore_prefixes,
                                            self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return AscendLinearMethod(self, origin_prefix,
                                      self.packed_modules_mapping)
        elif isinstance(layer, Attention) and \
            'fa_quant_type' in self.quant_description.keys() and \
            self.quant_description['fa_quant_type'] is not None:
            return AscendKVCacheMethod(self, origin_prefix)
        elif isinstance(layer, Attention) and self.quant_description.get(
                'kv_quant_type') == 'C8':
            return AscendKVCacheMethod(self, origin_prefix)
        elif isinstance(layer, FusedMoE):
            if self.should_ignore_layer(prefix, self.ignore_prefixes,
                                            self.packed_modules_mapping):
                return AscendUnquantizedFusedMoEMethod(layer.moe)
            return AscendFusedMoEMethod(self, prefix,
                                        self.packed_modules_mapping)
        elif isinstance(layer, VocabParallelEmbedding):
            if self.should_ignore_layer(prefix, self.ignore_prefixes,
                                            self.packed_modules_mapping):
                return UnquantizedEmbeddingMethod()
            return AscendEmbeddingMethod(self, origin_prefix,
                                         self.packed_modules_mapping)
        return None
    
    def _is_equal_or_regex_match(self, value: str,
                             target: str,
                             check_contains: bool = False) -> bool:
        """
        Checks whether a value is exactly equal or a regex match for target
        if target starts with 're:'. If check_contains is set to True,
        additionally checks if the target string is contained within the value.
        """

        if target.startswith("re:"):
            pattern = target[3:]
            if re.match(pattern, value):
                return True
        elif check_contains:
            if target.lower() in value.lower():
                return True
        elif target == value:
            return True
        return False
    
    def check_equal_or_regex_match(self, layer_name: str,
                               targets: Iterable[str] = tuple(),) -> bool:
        """
        Checks whether a layer_name is exactly equal or a regex match for
        if target starts with 're:' to any target in list.
        """
        for target in targets:
            if self._is_equal_or_regex_match(layer_name, target):
                return True
        return False
    
    def should_ignore_layer(self, layer_name: Optional[str], 
                            ignore: Iterable[str] = tuple(),
                            fused_mapping: Mapping[str, list[str]] = MappingProxyType({})) -> bool:
    
        if layer_name is None:
            return False

        # layer_name = model.layers.0.self_attn.qkv_proj
        # proj_name = qkv_proj
        proj_name = layer_name.split(".")[-1]

        # Fused layers like gate_up_proj or qkv_proj will not be fused
        # in the safetensors checkpoint. So, we convert the name
        # from the fused version to unfused + check to make sure that
        # each shard of the fused layer has the same scheme.
        if proj_name in fused_mapping and layer_name not in ignore:
            shard_proj_names = fused_mapping[proj_name]

            # Convert fused_name --> [shard_names]
            shard_names = [
                layer_name.replace(proj_name, shard_proj_name)
                for shard_proj_name in shard_proj_names
            ]

            # Layer should be ignored if shards are ignored.
            should_ignore_layer = None
            for shard_name in shard_names:
                should_ignore_shard =self.check_equal_or_regex_match(
                    layer_name=shard_name, targets=ignore)

                # If shard_idx=0, set layer ignore to match shard.
                if should_ignore_layer is None:
                    should_ignore_layer = should_ignore_shard

                # If shard_idx=1+ confirm scheme matches prior shards.
                elif should_ignore_shard != should_ignore_layer:
                    raise ValueError(
                        f"Detected some but not all shards of {proj_name} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision.")

        # Unfused layers like down_proj and o_proj will match
        # the safetensors checkpoint already.
        else:
            should_ignore_layer = self.check_equal_or_regex_match(layer_name=layer_name,
                                                            targets=ignore)

        assert should_ignore_layer is not None
        return should_ignore_layer

    def get_quant_method_modify(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        vllm_config = get_current_vllm_config()
        model_type = vllm_config.model_config.hf_config.model_type
        if model_type in packed_modules_model_mapping:
            self.packed_modules_mapping = packed_modules_model_mapping[
                model_type]
        from vllm.attention.layer import Attention
        if prefix.startswith("language_model"):
            prefix = prefix.split('.', 1)[-1]
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return AscendLinearMethod(self, prefix,
                                      self.packed_modules_mapping)
        elif isinstance(layer, Attention) and \
            'fa_quant_type' in self.quant_description.keys() and \
            self.quant_description['fa_quant_type'] is not None:
            return AscendKVCacheMethod(self, prefix)
        elif isinstance(layer, Attention) and self.quant_description.get(
                'kv_quant_type') == 'C8':
            return AscendKVCacheMethod(self, prefix)
        elif isinstance(layer, FusedMoE):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return AscendUnquantizedFusedMoEMethod(layer.moe_config)
            return AscendFusedMoEMethod(self, prefix,
                                        self.packed_modules_mapping)
        elif isinstance(layer, VocabParallelEmbedding):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return UnquantizedEmbeddingMethod()
            return AscendEmbeddingMethod(self, prefix,
                                         self.packed_modules_mapping)
        return None

    def is_layer_skipped_ascend(
        self,
        prefix: str,
        fused_mapping: Mapping[str, List[str]] = MappingProxyType({})):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = self.quant_description[shard_prefix +
                                                          '.weight'] == "FLOAT"

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision.")
        else:
            is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []


packed_modules_model_mapping = {
    "qwen3_moe": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    },
    "deepseek_v2": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    },
    "deepseek_v3": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    },
    # NOTE 1.The quantized MTP layer of deepseek on the NPU is not quantized;
    # NOTE 2.The description file generated by the current msmodelslim tool does not have
    # MTP layer info. Please manually add it and set the value to FLOAT.
    "deepseek_mtp": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    },
    "qwen3_next": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj": ["in_proj_qkvz", "in_proj_ba"],
    },
    "qwen2_5_vl": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    },
    "glm4_moe": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    },
}


class AscendLinearMethod(LinearMethodBase):
    """Linear method for Ascend quantization.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str,
                 packed_modules_mapping: Dict[str, Any]) -> None:
        self.quant_method = get_quant_method(quant_config.quant_description,
                                             prefix, "linear",
                                             packed_modules_mapping)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        weight_dict = self.quant_method.get_weight(input_size_per_partition,
                                                   output_size_per_partition,
                                                   params_dtype)
        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = torch.nn.Parameter(pertensor_param, requires_grad=False)
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)

        perchannel_dict = self.quant_method.get_perchannel_param(
            output_size_per_partition, params_dtype)
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pergroup_dict = self.quant_method.get_pergroup_param(
            input_size_per_partition, output_size_per_partition, params_dtype)
        for pergroup_name, pergroup_param in pergroup_dict.items():
            param = torch.nn.Parameter(pergroup_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(pergroup_name, param)
            set_weight_attrs(param, extra_weight_attrs)
            if "weight_scale_second" in pergroup_name or "weight_offset_second" in pergroup_name:
                setattr(param, "input_dim", 1)
                param.input_dim = 1

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(layer, RowParallelLinear):
            if layer.prefix.find("o_proj") != -1 and oproj_tp_enable():
                tp_rank = get_otp_group().rank_in_group
            elif layer.prefix.find("down_proj") != -1 and mlp_tp_enable():
                tp_rank = get_mlp_tp_group().rank_in_group
            else:
                tp_rank = get_tensor_model_parallel_rank()
        else:
            tp_rank = 0
        return self.quant_method.apply(layer, x, bias, tp_rank)


class AscendKVCacheMethod(BaseKVCacheMethod):
    """KVCache method for Ascend quantization.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str) -> None:
        self.quant_method = get_quant_method(quant_config.quant_description,
                                             prefix, "attention")

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Different from linear method, there are no weight processing/slicing
        # steps for attention in vllm. So the whole process of create weights
        # is hidden into the specific quant method.
        self.quant_method.create_weights(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(self, layer: torch.nn.Module, query: torch.Tensor,
              key: torch.Tensor, value: torch.Tensor, kv_cache, attn_metadata,
              attn_type, scale, output) -> torch.Tensor:
        return self.quant_method.apply(layer, query, key, value, kv_cache,
                                       attn_metadata, attn_type, scale, output)


class AscendFusedMoEMethod(FusedMoEMethodBase):
    """FusedMoE method for Ascend quantization.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str,
                 packed_modules_mapping: Dict[str, Any]):
        self.quant_method = get_quant_method(quant_config.quant_description,
                                             prefix, "moe",
                                             packed_modules_mapping)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_param = self.quant_method.get_weight(
            num_experts, intermediate_size_per_partition, hidden_size,
            params_dtype)
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        per_group_param = [
            "weight_scale_second", "weight_offset_second", "scale_bias"
        ]
        dynamic_quant_param = self.quant_method.get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_size,
            params_dtype)
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)
            if any(fields in param_key for fields in per_group_param):
                setattr(param, "quant_method",
                        FusedMoeWeightScaleSupported.GROUP.value)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num=0,
        **kwargs,
    ) -> torch.Tensor:
        return self.quant_method.apply(
            layer, x, router_logits, top_k, renormalize, use_grouped_topk,
            global_num_experts, expert_map, topk_group, num_expert_group,
            custom_routing_function, scoring_func, e_score_correction_bias,
            is_prefill, enable_force_load_balance, log2phy,
            global_redundant_expert_num, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        # TODO: implement this function
        pass


class AscendEmbeddingMethod(AscendLinearMethod):
    """Embedding method for Ascend quantization.
    
      Args:
          quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str,
                 packed_modules_mapping: Dict[str, Any]) -> None:
        self.quant_method = get_quant_method(quant_config.quant_description,
                                             prefix, "linear",
                                             packed_modules_mapping)
