import inspect
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vl_backend: str = "llava_video"
    vl_model_name: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    vl_dtype: str = "bfloat16"
    vl_max_text_len: int = 256
    freeze_vl: bool = False
    quantization_config: Optional[Any] = None
    num_labels: int = 174
    classifier_dropout: float = 0.1
    value_pooling: str = "hidden_mean"
    logits_to_keep: int = 0


class LLaVAVideoBackbone(nn.Module):
    @staticmethod
    def _normalize_media_size(image_size):
        if isinstance(image_size, (tuple, list)):
            if len(image_size) >= 2:
                return {"height": int(image_size[0]), "width": int(image_size[1])}
            if len(image_size) == 1:
                size = int(image_size[0])
                return {"height": size, "width": size}
        if image_size is None:
            return None
        size = int(image_size)
        return {"height": size, "width": size}

    def _configure_processor_media_size(self, cfg_hf):
        vision_cfg = getattr(cfg_hf, "vision_config", None)
        image_size = getattr(vision_cfg, "image_size", None) if vision_cfg is not None else None
        media_size = self._normalize_media_size(image_size)
        if media_size is None:
            return

        for proc_name in ("image_processor", "video_processor"):
            proc = getattr(self.processor, proc_name, None)
            if proc is None:
                continue
            if hasattr(proc, "size"):
                proc.size = dict(media_size)
            if hasattr(proc, "crop_size"):
                proc.crop_size = dict(media_size)

    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        if cfg.vl_dtype == "float16":
            dtype = torch.float16
        elif cfg.vl_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
            try:
                from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
            except Exception:
                AutoModelForImageTextToText = None
            try:
                from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
            except Exception:
                AutoModelForVision2Seq = None
            try:
                from transformers.models.llava_next_video import LlavaNextVideoForConditionalGeneration
            except Exception:
                LlavaNextVideoForConditionalGeneration = None
            try:
                from transformers.models.llava_onevision import LlavaOnevisionForConditionalGeneration
            except Exception:
                LlavaOnevisionForConditionalGeneration = None
        except Exception as e:
            raise ImportError("HF multimodal backbones require transformers installed.") from e

        trust_remote_code = cfg.vl_backend == "internvl"
        cfg_hf = AutoConfig.from_pretrained(cfg.vl_model_name, trust_remote_code=trust_remote_code)
        model_type = str(getattr(cfg_hf, "model_type", "")).lower()

        self.processor = AutoProcessor.from_pretrained(cfg.vl_model_name, trust_remote_code=trust_remote_code)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            cfg.vl_model_name,
            trust_remote_code=trust_remote_code,
        )
        self._configure_processor_media_size(cfg_hf)

        model_kwargs = {"torch_dtype": dtype}
        if cfg.quantization_config is not None:
            model_kwargs["quantization_config"] = cfg.quantization_config

        if cfg.vl_backend == "internvl":
            model_kwargs["trust_remote_code"] = True
            if AutoModelForImageTextToText is not None:
                self.model = AutoModelForImageTextToText.from_pretrained(cfg.vl_model_name, **model_kwargs)
            elif AutoModelForVision2Seq is not None:
                self.model = AutoModelForVision2Seq.from_pretrained(cfg.vl_model_name, **model_kwargs)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(cfg.vl_model_name, **model_kwargs)
        elif model_type == "llava_next_video":
            if LlavaNextVideoForConditionalGeneration is None:
                raise ImportError("This transformers build does not provide LlavaNextVideoForConditionalGeneration.")
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(cfg.vl_model_name, **model_kwargs)
        elif model_type == "llava_onevision":
            if LlavaOnevisionForConditionalGeneration is None:
                raise ImportError("This transformers build does not provide LlavaOnevisionForConditionalGeneration.")
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(cfg.vl_model_name, **model_kwargs)
        elif AutoModelForVision2Seq is not None:
            self.model = AutoModelForVision2Seq.from_pretrained(cfg.vl_model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.vl_model_name, **model_kwargs)

        self.model.to(device)
        if cfg.freeze_vl:
            for param in self.model.parameters():
                param.requires_grad = False

        self._dtype = dtype

    def _move_inputs_to_device(self, inputs):
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if key in ("pixel_values", "pixel_values_videos", "video_values", "video", "videos"):
                    moved[key] = value.to(self.device, dtype=self._dtype)
                else:
                    moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def get_input_embeddings(self):
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "get_input_embeddings"):
            return self.model.language_model.get_input_embeddings()
        if hasattr(self.model, "model") and hasattr(self.model.model, "get_input_embeddings"):
            return self.model.model.get_input_embeddings()
        raise AttributeError("Could not access input embeddings on the selected VLM backbone.")


class MultimodalBeliefModel(nn.Module):
    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.backbone = LLaVAVideoBackbone(cfg, device=device)
        try:
            self._backbone_forward_params = set(inspect.signature(self.backbone.model.forward).parameters.keys())
        except Exception:
            self._backbone_forward_params = set()

        feature_dim = self._infer_feature_dim()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(feature_dim, cfg.num_labels),
        )

    def _infer_vocab_size(self) -> int:
        get_out = getattr(self.backbone.model, "get_output_embeddings", None)
        if callable(get_out):
            try:
                out_emb = get_out()
            except Exception:
                out_emb = None
            if out_emb is not None and getattr(out_emb, "weight", None) is not None:
                return int(out_emb.weight.shape[0])

        cfg = getattr(self.backbone.model, "config", None)
        candidates = []
        if cfg is not None:
            candidates.append(getattr(cfg, "vocab_size", 0))
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                candidates.append(getattr(text_cfg, "vocab_size", 0))
        for candidate in candidates:
            try:
                value = int(candidate)
            except Exception:
                value = 0
            if value > 0:
                return value
        return 0

    def _infer_feature_dim(self) -> int:
        if self.cfg.value_pooling == "last_token_logits":
            vocab_size = self._infer_vocab_size()
            if vocab_size <= 0:
                raise RuntimeError("Unable to infer vocab size for logits-based pooling.")
            return vocab_size
        return int(self.backbone.get_input_embeddings().embedding_dim)

    def _pool_backbone_output(self, outputs, attention_mask):
        if self.cfg.value_pooling == "last_token_logits":
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("logits-based pooling requested but model output has no logits.")
            return logits[:, -1, :]

        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("hidden_mean pooling requested but hidden states were not returned.")
        last_hidden = hidden_states[-1]
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (last_hidden * mask).sum(dim=1) / denom

    def forward(self, inputs, labels=None, return_features=False):
        model_inputs = self.backbone._move_inputs_to_device(inputs)
        attention_mask = model_inputs.get("attention_mask")
        forward_kwargs = {"return_dict": True}
        if self.cfg.value_pooling == "hidden_mean":
            forward_kwargs["output_hidden_states"] = True
        if self.cfg.logits_to_keep > 0:
            if "logits_to_keep" in self._backbone_forward_params:
                forward_kwargs["logits_to_keep"] = self.cfg.logits_to_keep
            elif "num_logits_to_keep" in self._backbone_forward_params:
                forward_kwargs["num_logits_to_keep"] = self.cfg.logits_to_keep
        if hasattr(self.backbone.model, "config") and hasattr(self.backbone.model.config, "use_cache"):
            forward_kwargs["use_cache"] = False

        outputs = self.backbone.model(**model_inputs, **forward_kwargs)
        features = self._pool_backbone_output(outputs, attention_mask)
        features = features.to(dtype=self.classifier[2].weight.dtype, device=self.classifier[2].weight.device)
        logits = self.classifier(features)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        if return_features:
            result["features"] = features
        return result


MultimodalValueModel = MultimodalBeliefModel
