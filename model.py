from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vl_backend: str = "internvl"
    vl_model_name: str = "OpenGVLab/InternVL3_5-1B-HF"
    vl_dtype: str = "bfloat16"
    freeze_vl: bool = False
    quantization_config: Optional[Any] = None
    use_cache: bool = False


class InternVLBackbone(nn.Module):
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
        except Exception as e:
            raise ImportError("HF multimodal backbones require transformers installed.") from e

        if cfg.vl_backend != "internvl":
            raise RuntimeError(f"Unsupported vl_backend={cfg.vl_backend}. This repo now supports InternVL only.")

        trust_remote_code = True
        cfg_hf = AutoConfig.from_pretrained(cfg.vl_model_name, trust_remote_code=trust_remote_code)

        self.processor = AutoProcessor.from_pretrained(cfg.vl_model_name, trust_remote_code=trust_remote_code)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            cfg.vl_model_name,
            trust_remote_code=trust_remote_code,
        )
        self._configure_processor_media_size(cfg_hf)

        model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        if cfg.quantization_config is not None:
            model_kwargs["quantization_config"] = cfg.quantization_config

        if AutoModelForImageTextToText is not None:
            self.model = AutoModelForImageTextToText.from_pretrained(cfg.vl_model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.vl_model_name, **model_kwargs)

        self.model.to(device)
        if cfg.freeze_vl:
            for param in self.model.parameters():
                param.requires_grad = False
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = bool(cfg.use_cache)

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

    def _get_core_model(self):
        model = self.model
        if hasattr(model, "get_base_model"):
            try:
                model = model.get_base_model()
            except Exception:
                pass
        if hasattr(model, "model") and hasattr(model.model, "get_image_features"):
            model = model.model
        return model

    def build_pixel_values(self, frames):
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("InternVL processor does not expose an image_processor.")
        processed = image_processor(images=frames, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        return pixel_values.to(self.device, dtype=self._dtype)

    @torch.no_grad()
    def extract_frame_embeddings(self, frames, normalize: bool = True):
        pixel_values = self.build_pixel_values(frames)
        model_ref = self._get_core_model()
        image_features = model_ref.get_image_features(pixel_values=pixel_values)
        if not torch.is_tensor(image_features):
            if hasattr(image_features, "image_hidden_states") and image_features.image_hidden_states is not None:
                image_features = image_features.image_hidden_states
            elif hasattr(image_features, "last_hidden_state") and image_features.last_hidden_state is not None:
                image_features = image_features.last_hidden_state
            elif hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                image_features = image_features.pooler_output
            else:
                raise RuntimeError(f"Unsupported InternVL image feature return type: {type(image_features)}")
        if image_features.dim() == 3:
            frame_embeddings = image_features.mean(dim=1)
        elif image_features.dim() == 2:
            frame_embeddings = image_features
        else:
            raise RuntimeError(f"Unexpected image feature shape: {tuple(image_features.shape)}")
        if normalize:
            frame_embeddings = F.normalize(frame_embeddings.float(), dim=-1)
        return frame_embeddings

    @torch.no_grad()
    def extract_clip_embeddings(self, clips, normalize: bool = True):
        if not clips:
            raise RuntimeError("Expected at least one clip.")
        clip_lengths = [len(clip) for clip in clips]
        flat_frames = [frame for clip in clips for frame in clip]
        frame_embeddings = self.extract_frame_embeddings(flat_frames, normalize=normalize)
        splits = list(torch.split(frame_embeddings, clip_lengths, dim=0))
        return torch.stack(splits, dim=0)


class MultimodalVLMModel(nn.Module):
    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.backbone = InternVLBackbone(cfg, device=device)

    def encode_inputs(self, inputs, pooling: str = "last"):
        model_inputs = self.backbone._move_inputs_to_device(inputs)
        outputs = self.backbone.model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("The selected VLM backend did not return hidden states.")
        sequence_hidden = hidden_states[-1]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            pooled = sequence_hidden[:, -1, :]
        elif pooling == "mean":
            weights = attention_mask.to(sequence_hidden.device, dtype=sequence_hidden.dtype).unsqueeze(-1)
            pooled = (sequence_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            last_indices = attention_mask.sum(dim=1).long().clamp_min(1) - 1
            pooled = sequence_hidden[torch.arange(sequence_hidden.size(0), device=sequence_hidden.device), last_indices]
        return {
            "pooled_state": pooled,
            "sequence_hidden": sequence_hidden,
            "attention_mask": attention_mask,
        }

    def forward(self, inputs, labels=None):
        model_inputs = self.backbone._move_inputs_to_device(inputs)
        forward_kwargs = {"return_dict": True}
        if hasattr(self.backbone.model, "config") and hasattr(self.backbone.model.config, "use_cache"):
            forward_kwargs["use_cache"] = False
        if labels is not None:
            model_inputs["labels"] = labels.to(self.backbone.device)
        outputs = self.backbone.model(**model_inputs, **forward_kwargs)
        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens=64):
        model_inputs = self.backbone._move_inputs_to_device(inputs)
        return self.backbone.model.generate(**model_inputs, max_new_tokens=max_new_tokens)


MultimodalValueModel = MultimodalVLMModel
