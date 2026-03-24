import inspect
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_belief import BeliefProjector, RecurrentBeliefQueryModule
from future_prediction import FuturePredictionTransformer, FuturePredictorConfig


@dataclass
class ModelConfig:
    vl_backend: str = "internvl"
    vl_model_name: str = "OpenGVLab/InternVL3_5-1B-HF"
    vl_dtype: str = "bfloat16"
    freeze_vl: bool = False
    quantization_config: Optional[Any] = None
    use_cache: bool = False
    future_predictor_checkpoint: str = ""
    future_predictor_bundle: Optional[Any] = None
    future_context_frames: int = 0
    future_frames: int = 0
    use_belief_model: bool = False
    belief_num_heads: int = 8
    belief_num_tokens: int = 4
    belief_dropout: float = 0.1
    belief_hidden_dim: int = 1024
    belief_latent_dim: int = 512
    belief_num_layers: int = 2
    belief_target_frames: int = 2
    belief_max_text_tokens: int = 256
    belief_aux_weight: float = 0.2
    belief_future_weight: float = 1.0
    belief_reconstruction_weight: float = 0.5
    belief_kl_weight: float = 1e-3
    belief_fusion_scope: str = "vision_text"
    belief_use_recurrence: bool = True
    belief_temporal_chunks: int = 4
    disable_late_belief_prefix: bool = False


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
            raise RuntimeError(f"Unsupported vl_backend={cfg.vl_backend}. Belief-VLM now supports InternVL only.")

        trust_remote_code = True
        cfg_hf = AutoConfig.from_pretrained(cfg.vl_model_name, trust_remote_code=trust_remote_code)

        self.processor = AutoProcessor.from_pretrained(cfg.vl_model_name, trust_remote_code=trust_remote_code)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            cfg.vl_model_name,
            trust_remote_code=trust_remote_code,
        )
        self._configure_processor_media_size(cfg_hf)

        model_kwargs = {"torch_dtype": dtype}
        if cfg.quantization_config is not None:
            model_kwargs["quantization_config"] = cfg.quantization_config

        model_kwargs["trust_remote_code"] = True
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
                raise RuntimeError(
                    f"Unsupported InternVL image feature return type: {type(image_features)}"
                )
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

    def extract_image_tokens(self, pixel_values, normalize: bool = False):
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
                raise RuntimeError(
                    f"Unsupported InternVL image feature return type: {type(image_features)}"
                )
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        if normalize:
            image_features = F.normalize(image_features.float(), dim=-1)
        return image_features


class FutureTokenAdapter(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        delta = self.proj(self.norm(x))
        gate = torch.tanh(self.gate)
        return x + gate * delta


class BeliefTokenAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        proj_dtype = self.proj.weight.dtype
        x_in = x.to(dtype=proj_dtype)
        delta = self.proj(self.norm(x_in))
        gate = torch.tanh(self.gate)
        if x.shape[-1] == delta.shape[-1]:
            return x_in + gate * delta
        return gate * delta + (1.0 - gate) * self.proj.weight.new_zeros(delta.shape)


class MultimodalBeliefModel(nn.Module):
    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.backbone = InternVLBackbone(cfg, device=device)
        try:
            self._backbone_forward_params = set(inspect.signature(self.backbone.model.forward).parameters.keys())
        except Exception:
            self._backbone_forward_params = set()
        self.future_predictor = None
        self.future_adapter = None
        self.belief_predictor = None
        self.belief_adapter = None
        self.belief_to_vision_attn = None
        self.belief_to_text_attn = None
        self.vision_fusion_gate = None
        self.text_fusion_gate = None
        self.belief_summary_head = None
        self.choice_text_head = None
        self.image_token_adapter = None
        self._future_frames = int(cfg.future_frames or 0)
        self._future_context_frames = int(cfg.future_context_frames or 0)
        self._image_token_id = int(getattr(self.backbone.model.config, "image_token_id", -1))
        token_embed = self.backbone.model.get_input_embeddings()
        self._lm_embed_dim = int(getattr(token_embed, "embedding_dim", 0) or 0)
        self._vision_feature_dim = int(self._infer_vision_feature_dim() or 0)
        if self._vision_feature_dim > 0 and (
            cfg.use_belief_model or cfg.future_predictor_bundle is not None or bool(cfg.future_predictor_checkpoint)
        ):
            self._ensure_feature_adapters(self._vision_feature_dim)
        if cfg.future_predictor_bundle is not None:
            self._init_future_conditioning_from_bundle(cfg.future_predictor_bundle)
        elif cfg.future_predictor_checkpoint:
            self._init_future_conditioning(cfg.future_predictor_checkpoint)
        if cfg.use_belief_model:
            self._init_belief_conditioning()

    def _infer_vision_feature_dim(self):
        model_ref = self.backbone._get_core_model()
        for attr in ("projection_dim", "hidden_size", "embed_dim"):
            value = getattr(getattr(model_ref, "vision_model", model_ref), "config", None)
            if value is not None and hasattr(value, attr):
                return int(getattr(value, attr))
        image_processor = getattr(self.backbone.processor, "image_processor", None)
        if image_processor is not None:
            size = getattr(image_processor, "size", None) or {}
            height = int(size.get("height", size.get("shortest_edge", 448)) or 448)
            width = int(size.get("width", size.get("shortest_edge", height)) or height)
        else:
            height = 448
            width = 448
        try:
            with torch.no_grad():
                dummy_pixels = torch.zeros(
                    1,
                    3,
                    height,
                    width,
                    device=self.backbone.device,
                    dtype=self.backbone._dtype,
                )
                image_features = model_ref.get_image_features(pixel_values=dummy_pixels)
                if not torch.is_tensor(image_features):
                    if hasattr(image_features, "image_hidden_states") and image_features.image_hidden_states is not None:
                        image_features = image_features.image_hidden_states
                    elif hasattr(image_features, "last_hidden_state") and image_features.last_hidden_state is not None:
                        image_features = image_features.last_hidden_state
                    elif hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                        image_features = image_features.pooler_output
                if torch.is_tensor(image_features):
                    return int(image_features.shape[-1])
        except Exception:
            pass
        return 0

    def _ensure_feature_adapters(self, image_feature_dim: int):
        if self._lm_embed_dim <= 0:
            raise RuntimeError("Could not infer the LM embedding size for conditioned inputs.")
        if image_feature_dim <= 0:
            raise RuntimeError("Could not infer the visual feature size for conditioned inputs.")
        if image_feature_dim != self._lm_embed_dim and self.image_token_adapter is None:
            self.image_token_adapter = nn.Linear(image_feature_dim, self._lm_embed_dim).to(self.backbone.device)

    @staticmethod
    def _resize_token_sequence(tokens: torch.Tensor, target_tokens: int):
        current_tokens = int(tokens.shape[1])
        if current_tokens == target_tokens:
            return tokens
        if target_tokens <= 0:
            raise RuntimeError(f"target_tokens must be positive, got {target_tokens}")
        resized = F.interpolate(
            tokens.transpose(1, 2),
            size=target_tokens,
            mode="linear",
            align_corners=False,
        )
        return resized.transpose(1, 2)

    def _init_belief_conditioning(self):
        if self._lm_embed_dim <= 0:
            raise RuntimeError("Could not infer the LM embedding size for belief conditioning.")
        self.belief_predictor = RecurrentBeliefQueryModule(
            d_model=self._lm_embed_dim,
            num_queries=int(self.cfg.belief_num_tokens),
            num_heads=int(self.cfg.belief_num_heads),
            dropout=float(self.cfg.belief_dropout),
            use_text_conditioning=True,
        ).to(self.backbone.device)
        self.belief_adapter = BeliefProjector(self._lm_embed_dim, self._lm_embed_dim).to(self.backbone.device)
        self.belief_to_vision_attn = nn.MultiheadAttention(
            embed_dim=self._lm_embed_dim,
            num_heads=int(self.cfg.belief_num_heads),
            dropout=float(self.cfg.belief_dropout),
            batch_first=True,
        ).to(self.backbone.device)
        self.belief_to_text_attn = nn.MultiheadAttention(
            embed_dim=self._lm_embed_dim,
            num_heads=int(self.cfg.belief_num_heads),
            dropout=float(self.cfg.belief_dropout),
            batch_first=True,
        ).to(self.backbone.device)
        self.vision_fusion_gate = nn.Parameter(torch.tensor(0.0, device=self.backbone.device))
        self.text_fusion_gate = nn.Parameter(torch.tensor(0.0, device=self.backbone.device))
        self.belief_summary_head = nn.Linear(self._lm_embed_dim, self._lm_embed_dim).to(self.backbone.device)
        self.choice_text_head = nn.Linear(self._lm_embed_dim, self._lm_embed_dim).to(self.backbone.device)

    def _build_future_predictor_from_state(self, predictor_state, saved_args):
        embed_dim = int(saved_args["predictor_embed_dim"]) if "predictor_embed_dim" in saved_args else int(saved_args.get("embed_dim", 0))
        if embed_dim <= 0:
            input_proj_weight = predictor_state.get("input_proj.weight")
            output_proj_weight = predictor_state.get("output_proj.weight")
            if input_proj_weight is not None:
                embed_dim = int(input_proj_weight.shape[1])
            elif output_proj_weight is not None:
                embed_dim = int(output_proj_weight.shape[0])
        max_context_frames = int(saved_args.get("video_frames", self._future_context_frames or 0))
        if max_context_frames <= 0:
            context_pos_embed = predictor_state.get("context_pos_embed")
            if context_pos_embed is not None:
                max_context_frames = int(context_pos_embed.shape[1])
        max_future_frames = int(saved_args.get("future_frames", self._future_frames or 0))
        if max_future_frames <= 0:
            future_queries = predictor_state.get("future_queries")
            if future_queries is not None:
                max_future_frames = int(future_queries.shape[1])
        predictor_cfg = FuturePredictorConfig(
            embed_dim=embed_dim,
            hidden_dim=int(saved_args.get("predictor_hidden_dim", 1024)),
            num_layers=int(saved_args.get("predictor_layers", 2)),
            num_heads=int(saved_args.get("predictor_heads", 8)),
            dropout=float(saved_args.get("predictor_dropout", 0.1)),
            max_context_frames=max_context_frames or 8,
            max_future_frames=max_future_frames or 8,
        )
        if predictor_cfg.embed_dim <= 0:
            raise RuntimeError("Future predictor checkpoint is missing embed-dimension metadata.")
        predictor = FuturePredictionTransformer(predictor_cfg)
        predictor.load_state_dict(predictor_state)
        predictor.to(self.backbone.device)
        predictor.eval()
        for param in predictor.parameters():
            param.requires_grad = False
        self.future_predictor = predictor
        self.future_adapter = BeliefTokenAdapter(predictor_cfg.embed_dim, self._lm_embed_dim).to(self.backbone.device)
        self._future_context_frames = int(self._future_context_frames or predictor_cfg.max_context_frames)
        self._future_frames = int(self._future_frames or predictor_cfg.max_future_frames)

    def _init_future_conditioning(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        saved_args = ckpt.get("args", {})
        predictor_state = ckpt.get("predictor", {})
        self._build_future_predictor_from_state(predictor_state, saved_args)

    def _init_future_conditioning_from_bundle(self, bundle):
        saved_args = bundle.get("args", {})
        predictor_state = bundle.get("predictor", {})
        self._build_future_predictor_from_state(predictor_state, saved_args)

    def export_future_predictor_bundle(self):
        if self.future_predictor is None:
            return None
        predictor = self.future_predictor
        return {
            "predictor": predictor.state_dict(),
            "args": {
                "predictor_embed_dim": int(predictor.cfg.embed_dim),
                "predictor_hidden_dim": int(predictor.cfg.hidden_dim),
                "predictor_layers": int(predictor.cfg.num_layers),
                "predictor_heads": int(predictor.cfg.num_heads),
                "predictor_dropout": float(predictor.cfg.dropout),
                "video_frames": int(predictor.cfg.max_context_frames),
                "future_frames": int(predictor.cfg.max_future_frames),
            },
        }

    def _compose_conditioned_inputs(
        self,
        input_ids,
        attention_mask,
        labels,
        image_features,
        prompt_override_embeddings=None,
        prompt_positions=None,
        belief_features=None,
        insert_belief_prefix=False,
    ):
        batch_size = image_features.shape[0]
        token_embed = self.backbone.model.get_input_embeddings()
        row_embeds = []
        row_attention_masks = []
        row_labels_out = [] if labels is not None else None

        for row in range(batch_size):
            row_ids = input_ids[row]
            row_attn = attention_mask[row]
            valid_len = int(row_attn.sum().item())
            row_ids = row_ids[:valid_len]
            row_labels = labels[row][:valid_len] if labels is not None else None
            row_embeds_cur = token_embed(row_ids)

            image_mask = row_ids.eq(self._image_token_id)
            image_token_count = int(image_mask.sum().item())
            row_image_features = image_features[row]
            if image_token_count != int(row_image_features.shape[0]):
                row_image_features = self._resize_token_sequence(
                    row_image_features.unsqueeze(0),
                    image_token_count,
                ).squeeze(0)
            if image_token_count > 0:
                row_embeds_cur = row_embeds_cur.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(row_embeds_cur),
                    row_image_features.to(row_embeds_cur.device, row_embeds_cur.dtype),
                )

            if prompt_override_embeddings is not None and prompt_positions is not None:
                positions = prompt_positions[row]
                positions = positions[positions.ge(0)]
                if positions.numel() > 0:
                    row_prompt = prompt_override_embeddings[row][: positions.numel()].to(
                        row_embeds_cur.device, row_embeds_cur.dtype
                    )
                    row_embeds_cur[positions] = row_prompt

            belief_count = 0
            if insert_belief_prefix and belief_features is not None:
                if row_labels is not None:
                    non_mask = torch.nonzero(row_labels.ne(-100), as_tuple=False)
                    insert_at = int(non_mask[0].item()) if non_mask.numel() > 0 else valid_len
                else:
                    insert_at = valid_len
                belief_row = belief_features[row].to(row_embeds_cur.device, row_embeds_cur.dtype)
                row_embeds_cur = torch.cat(
                    [row_embeds_cur[:insert_at], belief_row, row_embeds_cur[insert_at:]],
                    dim=0,
                )
                belief_count = int(belief_row.shape[0])

            row_embeds.append(row_embeds_cur)
            row_attention_masks.append(
                torch.ones(row_embeds_cur.shape[0], dtype=attention_mask.dtype, device=attention_mask.device)
            )
            if row_labels is not None:
                if belief_count > 0:
                    added_labels = torch.full((belief_count,), -100, dtype=row_labels.dtype, device=row_labels.device)
                    row_labels = torch.cat([row_labels[:insert_at], added_labels, row_labels[insert_at:]], dim=0)
                row_labels_out.append(row_labels)

        max_len = max(int(row.shape[0]) for row in row_embeds)
        padded_embeds = []
        padded_attn = []
        padded_labels = [] if row_labels_out is not None else None
        for idx in range(batch_size):
            embeds = row_embeds[idx]
            pad_len = max_len - int(embeds.shape[0])
            if pad_len > 0:
                embeds = F.pad(embeds, (0, 0, 0, pad_len), value=0.0)
            padded_embeds.append(embeds)
            padded_attn.append(F.pad(row_attention_masks[idx], (0, pad_len), value=0))
            if row_labels_out is not None:
                padded_labels.append(F.pad(row_labels_out[idx], (0, pad_len), value=-100))

        inputs_embeds = torch.stack(padded_embeds, dim=0)
        new_attention_mask = torch.stack(padded_attn, dim=0)
        new_labels = torch.stack(padded_labels, dim=0) if padded_labels is not None else None
        return inputs_embeds, new_attention_mask, new_labels

    def _extract_frame_context(self, model_inputs):
        pixel_key = None
        for candidate in ("pixel_values", "pixel_values_videos", "video_values", "video", "videos"):
            if candidate in model_inputs:
                pixel_key = candidate
                break
        if pixel_key is None:
            raise RuntimeError("Conditioned training requires pixel values in the model inputs.")
        pixel_values = model_inputs[pixel_key]
        image_features = self.backbone.extract_image_tokens(pixel_values, normalize=False)
        batch_size = model_inputs["input_ids"].shape[0]
        num_images = image_features.shape[0]
        if num_images % batch_size != 0:
            raise RuntimeError(
                f"Could not reshape image features by batch: num_images={num_images}, batch={batch_size}"
            )
        frames_per_sample = num_images // batch_size
        frame_embeddings = image_features.mean(dim=1).view(batch_size, frames_per_sample, -1)
        self._ensure_feature_adapters(int(image_features.shape[-1]))
        if self.image_token_adapter is not None:
            adapter_dtype = self.image_token_adapter.weight.dtype
            image_features = self.image_token_adapter(image_features.to(dtype=adapter_dtype))
        image_tokens = image_features.view(batch_size, -1, image_features.shape[-1])
        return image_tokens, frame_embeddings, frames_per_sample

    def _extract_prompt_text_context(self, model_inputs, labels=None):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        token_embed = self.backbone.model.get_input_embeddings()
        text_rows = []
        max_text_tokens = 0
        position_rows = []

        for row in range(input_ids.shape[0]):
            row_ids = input_ids[row]
            valid_len = int(attention_mask[row].sum().item())
            row_ids = row_ids[:valid_len]
            prompt_mask = torch.ones(valid_len, dtype=torch.bool, device=row_ids.device)
            if labels is not None:
                row_labels = labels[row][:valid_len]
                prompt_mask = row_labels.eq(-100)
            text_mask = prompt_mask & row_ids.ne(self._image_token_id)
            selected_ids = row_ids[text_mask]
            if selected_ids.numel() == 0:
                row_text = token_embed.weight.new_zeros((0, token_embed.embedding_dim))
            else:
                row_text = token_embed(selected_ids)
            text_rows.append(row_text)
            max_text_tokens = max(max_text_tokens, int(row_text.shape[0]))
            position_rows.append(torch.nonzero(text_mask, as_tuple=False).squeeze(-1))

        if max_text_tokens == 0:
            return None, None, None

        padded_rows = []
        padding_masks = []
        padded_positions = []
        for row_text in text_rows:
            pad_len = max_text_tokens - int(row_text.shape[0])
            if pad_len > 0:
                row_text = F.pad(row_text, (0, 0, 0, pad_len), value=0.0)
            padded_rows.append(row_text)
            padding_mask = torch.zeros(max_text_tokens, dtype=torch.bool, device=row_text.device)
            if pad_len > 0:
                padding_mask[-pad_len:] = True
            padding_masks.append(padding_mask)
        for positions in position_rows:
            pad_len = max_text_tokens - int(positions.shape[0])
            if pad_len > 0:
                positions = F.pad(positions, (0, pad_len), value=-1)
            padded_positions.append(positions)

        return (
            torch.stack(padded_rows, dim=0),
            torch.stack(padding_masks, dim=0),
            torch.stack(padded_positions, dim=0),
        )

    def _extract_answer_text_context(self, model_inputs, labels):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        token_embed = self.backbone.model.get_input_embeddings()
        text_rows = []
        max_text_tokens = 0

        for row in range(input_ids.shape[0]):
            row_ids = input_ids[row]
            valid_len = int(attention_mask[row].sum().item())
            row_ids = row_ids[:valid_len]
            row_labels = labels[row][:valid_len]
            answer_mask = row_labels.ne(-100) & row_ids.ne(self._image_token_id)
            selected_ids = row_ids[answer_mask]
            if selected_ids.numel() == 0:
                row_text = token_embed.weight.new_zeros((0, token_embed.embedding_dim))
            else:
                row_text = token_embed(selected_ids)
            text_rows.append(row_text)
            max_text_tokens = max(max_text_tokens, int(row_text.shape[0]))

        if max_text_tokens == 0:
            return None, None

        padded_rows = []
        padding_masks = []
        for row_text in text_rows:
            pad_len = max_text_tokens - int(row_text.shape[0])
            if pad_len > 0:
                row_text = F.pad(row_text, (0, 0, 0, pad_len), value=0.0)
            padded_rows.append(row_text)
            padding_mask = torch.zeros(max_text_tokens, dtype=torch.bool, device=row_text.device)
            if pad_len > 0:
                padding_mask[-pad_len:] = True
            padding_masks.append(padding_mask)
        return torch.stack(padded_rows, dim=0), torch.stack(padding_masks, dim=0)

    def _build_belief_state(self, image_tokens, text_embeddings, text_padding_mask, frames_per_sample):
        predictor_dtype = next(self.belief_predictor.parameters()).dtype
        projector_dtype = next(self.belief_adapter.parameters()).dtype
        vision_tokens = image_tokens.to(self.backbone.device, dtype=predictor_dtype)
        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(self.backbone.device, dtype=predictor_dtype)
            text_padding_mask = text_padding_mask.to(self.backbone.device) if text_padding_mask is not None else None
        else:
            text_embeddings = vision_tokens.new_zeros((vision_tokens.shape[0], 1, vision_tokens.shape[-1]))
            text_padding_mask = torch.zeros(vision_tokens.shape[0], 1, dtype=torch.bool, device=self.backbone.device)

        tokens_per_frame = max(1, int(vision_tokens.shape[1] // max(frames_per_sample, 1)))
        vision_per_frame = vision_tokens.view(vision_tokens.shape[0], frames_per_sample, tokens_per_frame, -1)
        chunk_count = max(1, int(self.cfg.belief_temporal_chunks))
        chunk_size = max(1, (frames_per_sample + chunk_count - 1) // chunk_count)
        prev_belief = None
        belief_outputs = []
        for start in range(0, frames_per_sample, chunk_size):
            end = min(frames_per_sample, start + chunk_size)
            chunk_tokens = vision_per_frame[:, start:end].reshape(vision_tokens.shape[0], -1, vision_tokens.shape[-1])
            vision_padding_mask = torch.zeros(
                chunk_tokens.shape[0],
                chunk_tokens.shape[1],
                dtype=torch.bool,
                device=self.backbone.device,
            )
            key_padding_mask = torch.cat([vision_padding_mask, text_padding_mask], dim=1)
            prev_belief, _ = self.belief_predictor(
                Z_v=chunk_tokens,
                Z_x=text_embeddings,
                prev_belief=prev_belief if self.cfg.belief_use_recurrence else None,
                key_padding_mask=key_padding_mask,
            )
            belief_outputs.append(prev_belief)
        belief_tokens = belief_outputs[-1]
        belief_tokens = self.belief_adapter(belief_tokens.to(dtype=projector_dtype)).to(dtype=image_tokens.dtype)
        belief_summary = belief_tokens.mean(dim=1)
        return belief_tokens, belief_summary

    def _apply_belief_fusion(self, image_tokens, prompt_text_embeddings, prompt_padding_mask, belief_tokens):
        fused_image_tokens = image_tokens
        fused_prompt_embeddings = prompt_text_embeddings
        scope = self.cfg.belief_fusion_scope
        if scope in {"vision_only", "vision_text"}:
            delta_v, _ = self.belief_to_vision_attn(
                query=image_tokens,
                key=belief_tokens,
                value=belief_tokens,
            )
            fused_image_tokens = image_tokens + torch.tanh(self.vision_fusion_gate) * delta_v
        if scope == "vision_text" and prompt_text_embeddings is not None:
            text_query = prompt_text_embeddings
            delta_t, _ = self.belief_to_text_attn(
                query=text_query,
                key=belief_tokens,
                value=belief_tokens,
                key_padding_mask=None,
            )
            if prompt_padding_mask is not None:
                delta_t = delta_t.masked_fill(prompt_padding_mask.unsqueeze(-1), 0.0)
            fused_prompt_embeddings = text_query + torch.tanh(self.text_fusion_gate) * delta_t
        return fused_image_tokens, fused_prompt_embeddings

    def _forward_with_future(self, model_inputs, labels, forward_kwargs):
        image_tokens, context_frame_embeddings, _ = self._extract_frame_context(model_inputs)
        predictor_dtype = next(self.future_predictor.parameters()).dtype
        adapter_dtype = next(self.future_adapter.parameters()).dtype
        context_frame_embeddings = context_frame_embeddings.to(dtype=predictor_dtype)
        future_pred = self.future_predictor(context_frame_embeddings, future_frames=self._future_frames)
        future_tokens = self.future_adapter(future_pred.to(dtype=adapter_dtype))
        future_tokens = future_tokens.to(dtype=image_tokens.dtype)
        inputs_embeds, new_attention_mask, new_labels = self._compose_conditioned_inputs(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=labels,
            image_features=image_tokens,
            belief_features=future_tokens,
            insert_belief_prefix=True,
        )
        outputs = self.backbone.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **forward_kwargs,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    def _forward_with_belief(self, model_inputs, labels, forward_kwargs, return_aux=False):
        image_tokens, _, frames_per_sample = self._extract_frame_context(model_inputs)
        text_embeddings, text_padding_mask, prompt_positions = self._extract_prompt_text_context(model_inputs, labels=labels)
        belief_tokens, belief_summary = self._build_belief_state(
            image_tokens=image_tokens,
            text_embeddings=text_embeddings,
            text_padding_mask=text_padding_mask,
            frames_per_sample=frames_per_sample,
        )
        fused_image_tokens = image_tokens
        fused_prompt_embeddings = text_embeddings
        insert_belief_prefix = self.cfg.belief_fusion_scope == "late_prefix" and not self.cfg.disable_late_belief_prefix
        if self.cfg.belief_fusion_scope in {"vision_only", "vision_text"}:
            fused_image_tokens, fused_prompt_embeddings = self._apply_belief_fusion(
                image_tokens=image_tokens,
                prompt_text_embeddings=text_embeddings,
                prompt_padding_mask=text_padding_mask,
                belief_tokens=belief_tokens,
            )
            insert_belief_prefix = False

        inputs_embeds, new_attention_mask, new_labels = self._compose_conditioned_inputs(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=labels,
            image_features=fused_image_tokens,
            prompt_override_embeddings=fused_prompt_embeddings,
            prompt_positions=prompt_positions,
            belief_features=belief_tokens,
            insert_belief_prefix=insert_belief_prefix,
        )
        outputs = self.backbone.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **forward_kwargs,
        )
        result = {"loss": outputs.loss, "logits": outputs.logits}
        if return_aux:
            result["belief_tokens"] = belief_tokens
            result["belief_summary"] = self.belief_summary_head(belief_summary)
            if labels is not None:
                answer_embeddings, answer_padding_mask = self._extract_answer_text_context(model_inputs, labels)
                if answer_embeddings is not None:
                    valid_mask = (~answer_padding_mask).unsqueeze(-1)
                    pooled_answers = (answer_embeddings * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
                    result["choice_text_summary"] = self.choice_text_head(pooled_answers)
        return result

    def forward(self, inputs, labels=None, return_aux=False):
        model_inputs = self.backbone._move_inputs_to_device(inputs)
        forward_kwargs = {"return_dict": True}
        if hasattr(self.backbone.model, "config") and hasattr(self.backbone.model.config, "use_cache"):
            forward_kwargs["use_cache"] = False
        labels = labels.to(self.backbone.device) if labels is not None else None
        if self.future_predictor is not None:
            return self._forward_with_future(model_inputs, labels, forward_kwargs)
        if self.belief_predictor is not None:
            return self._forward_with_belief(model_inputs, labels, forward_kwargs, return_aux=return_aux)
        if labels is not None:
            model_inputs["labels"] = labels
        outputs = self.backbone.model(**model_inputs, **forward_kwargs)
        return {"loss": outputs.loss, "logits": outputs.logits}

    def _build_conditioned_generation_inputs(self, model_inputs):
        image_tokens, frame_embeddings, frames_per_sample = self._extract_frame_context(model_inputs)
        belief_features = None
        prompt_embeddings = None
        prompt_padding_mask = None
        prompt_positions = None
        if self.future_predictor is not None:
            predictor_dtype = next(self.future_predictor.parameters()).dtype
            adapter_dtype = next(self.future_adapter.parameters()).dtype
            future_pred = self.future_predictor(
                frame_embeddings.to(dtype=predictor_dtype),
                future_frames=self._future_frames,
            )
            belief_features = self.future_adapter(future_pred.to(dtype=adapter_dtype)).to(dtype=image_tokens.dtype)
        elif self.belief_predictor is not None:
            prompt_embeddings, prompt_padding_mask, prompt_positions = self._extract_prompt_text_context(
                model_inputs, labels=None
            )
            belief_features, _ = self._build_belief_state(
                image_tokens=image_tokens,
                text_embeddings=prompt_embeddings,
                text_padding_mask=prompt_padding_mask,
                frames_per_sample=frames_per_sample,
            )
            if self.cfg.belief_fusion_scope in {"vision_only", "vision_text"}:
                image_tokens, prompt_embeddings = self._apply_belief_fusion(
                    image_tokens=image_tokens,
                    prompt_text_embeddings=prompt_embeddings,
                    prompt_padding_mask=prompt_padding_mask,
                    belief_tokens=belief_features,
                )
        inputs_embeds, new_attention_mask, _ = self._compose_conditioned_inputs(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=None,
            image_features=image_tokens,
            prompt_override_embeddings=prompt_embeddings,
            prompt_positions=prompt_positions,
            belief_features=belief_features,
            insert_belief_prefix=self.cfg.belief_fusion_scope == "late_prefix" and not self.cfg.disable_late_belief_prefix,
        )
        return {"inputs_embeds": inputs_embeds, "attention_mask": new_attention_mask}

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens=64):
        model_inputs = self.backbone._move_inputs_to_device(inputs)
        if self.future_predictor is None and self.belief_predictor is None:
            return self.backbone.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        conditioned_inputs = self._build_conditioned_generation_inputs(model_inputs)
        return self.backbone.model.generate(**conditioned_inputs, max_new_tokens=max_new_tokens)


MultimodalValueModel = MultimodalBeliefModel
