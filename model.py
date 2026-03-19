import inspect
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Belief_Network import AttentionConfig, RecursiveBeliefNetwork
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
    belief_network_checkpoint: str = ""
    belief_network_bundle: Optional[Any] = None
    train_belief_network: bool = False
    belief_aux_weight: float = 0.0
    future_context_frames: int = 0
    future_frames: int = 0


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
        self.belief_network = None
        self.belief_adapter = None
        self._train_belief_network = bool(cfg.train_belief_network)
        self._belief_aux_weight = float(cfg.belief_aux_weight)
        self._future_frames = int(cfg.future_frames or 0)
        self._future_context_frames = int(cfg.future_context_frames or 0)
        self._image_token_id = int(getattr(self.backbone.model.config, "image_token_id", -1))
        if cfg.future_predictor_checkpoint and cfg.belief_network_checkpoint:
            raise RuntimeError("Configure only one of future_predictor_checkpoint or belief_network_checkpoint.")
        if cfg.future_predictor_bundle is not None:
            self._init_future_conditioning_from_bundle(cfg.future_predictor_bundle)
        elif cfg.future_predictor_checkpoint:
            self._init_future_conditioning(cfg.future_predictor_checkpoint)
        elif cfg.belief_network_bundle is not None:
            self._init_belief_conditioning_from_bundle(cfg.belief_network_bundle)
        elif cfg.belief_network_checkpoint:
            self._init_belief_conditioning(cfg.belief_network_checkpoint)

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
        self.future_adapter = FutureTokenAdapter(predictor_cfg.embed_dim).to(self.backbone.device)
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

    def _build_belief_network_from_state(self, belief_state, saved_args):
        visual_dim = int(saved_args.get("visual_dim", 0))
        if visual_dim <= 0:
            visual_proj_weight = belief_state.get("visual_proj.weight")
            if visual_proj_weight is not None:
                visual_dim = int(visual_proj_weight.shape[1])
        if visual_dim <= 0:
            raise RuntimeError("Belief network checkpoint is missing visual-dimension metadata.")
        belief_cfg = AttentionConfig(
            num_attention_heads=int(saved_args.get("num_attention_heads", 12)),
            num_hidden_layers=int(saved_args.get("num_hidden_layers", 12)),
            state_dim=int(saved_args.get("state_dim", 12)),
            belief_dim=int(saved_args.get("belief_dim", 3)),
            hidden_size=int(saved_args.get("hidden_size", 768)),
            latent_dim=int(saved_args.get("latent_dim", 32)),
            attention_dropout=float(saved_args.get("attention_dropout", 0.3)),
            proj_dropout=float(saved_args.get("proj_dropout", 0.2)),
        )
        belief_net = RecursiveBeliefNetwork(
            config=belief_cfg,
            visual_dim=visual_dim,
            beta=float(saved_args.get("beta", 1.0)),
            recon_weight=float(saved_args.get("recon_weight", 1.0)),
            temporal_nce_weight=float(saved_args.get("temporal_nce_weight", 0.0)),
            device=self.backbone.device,
        )
        belief_net.load_state_dict(belief_state)
        belief_net.to(self.backbone.device)
        if self._train_belief_network:
            belief_net.train()
        else:
            belief_net.eval()
            for param in belief_net.parameters():
                param.requires_grad = False
        self.belief_network = belief_net
        self.belief_adapter = FutureTokenAdapter(visual_dim).to(self.backbone.device)

    def _init_belief_conditioning(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        saved_args = ckpt.get("args", {})
        belief_state = ckpt.get("belief_net", {})
        self._build_belief_network_from_state(belief_state, saved_args)

    def _init_belief_conditioning_from_bundle(self, bundle):
        saved_args = bundle.get("args", {})
        belief_state = bundle.get("belief_net", {})
        self._build_belief_network_from_state(belief_state, saved_args)

    def export_belief_network_bundle(self):
        if self.belief_network is None:
            return None
        belief = self.belief_network
        cfg = belief.config
        return {
            "belief_net": belief.state_dict(),
            "args": {
                "visual_dim": int(belief.visual_dim),
                "state_dim": int(cfg.state_dim),
                "belief_dim": int(cfg.belief_dim),
                "hidden_size": int(cfg.hidden_size),
                "latent_dim": int(cfg.latent_dim),
                "num_attention_heads": int(cfg.num_attention_heads),
                "num_hidden_layers": int(cfg.num_hidden_layers),
                "attention_dropout": float(cfg.attention_dropout),
                "proj_dropout": float(cfg.proj_dropout),
                "beta": float(getattr(belief, "beta", 1.0)),
                "recon_weight": float(getattr(belief, "recon_weight", 1.0)),
                "temporal_nce_weight": float(getattr(belief, "temporal_nce_weight", 0.0)),
            },
        }

    def _inject_image_features(self, input_ids, attention_mask, labels, image_features, extra_features=None):
        batch_size, _, hidden_dim = image_features.shape
        token_embed = self.backbone.model.get_input_embeddings()
        new_input_ids = []
        new_attention_masks = []
        new_labels = [] if labels is not None else None
        flat_features = []

        for row in range(batch_size):
            row_ids = input_ids[row]
            row_attn = attention_mask[row]
            valid_len = int(row_attn.sum().item())
            row_ids = row_ids[:valid_len]
            row_attn = row_attn[:valid_len]
            row_labels = labels[row][:valid_len] if labels is not None else None

            if row_labels is not None:
                non_mask = torch.nonzero(row_labels.ne(-100), as_tuple=False)
                insert_at = int(non_mask[0].item()) if non_mask.numel() > 0 else valid_len
            else:
                insert_at = valid_len

            extra_count = 0 if extra_features is None else int(extra_features.shape[1])
            added_ids = torch.full((extra_count,), self._image_token_id, dtype=row_ids.dtype, device=row_ids.device)
            new_row_ids = torch.cat([row_ids[:insert_at], added_ids, row_ids[insert_at:]], dim=0)
            new_row_attn = torch.ones_like(new_row_ids)
            new_input_ids.append(new_row_ids)
            new_attention_masks.append(new_row_attn)

            row_features = image_features[row]
            if extra_features is not None:
                row_features = torch.cat([row_features, extra_features[row]], dim=0)
            flat_features.append(row_features)

            if row_labels is not None:
                added_labels = torch.full((extra_count,), -100, dtype=row_labels.dtype, device=row_labels.device)
                new_row_labels = torch.cat([row_labels[:insert_at], added_labels, row_labels[insert_at:]], dim=0)
                new_labels.append(new_row_labels)

        max_len = max(int(row.shape[0]) for row in new_input_ids)
        pad_id = self.backbone.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.backbone.tokenizer.eos_token_id
        padded_ids = []
        padded_attn = []
        padded_labels = [] if new_labels is not None else None
        for idx in range(batch_size):
            row_ids = new_input_ids[idx]
            row_attn = new_attention_masks[idx]
            pad_len = max_len - int(row_ids.shape[0])
            padded_ids.append(F.pad(row_ids, (0, pad_len), value=pad_id))
            padded_attn.append(F.pad(row_attn, (0, pad_len), value=0))
            if new_labels is not None:
                padded_labels.append(F.pad(new_labels[idx], (0, pad_len), value=-100))

        new_input_ids = torch.stack(padded_ids, dim=0)
        new_attention_mask = torch.stack(padded_attn, dim=0)
        new_labels = torch.stack(padded_labels, dim=0) if padded_labels is not None else None
        inputs_embeds = token_embed(new_input_ids)
        special_image_mask = new_input_ids.eq(self._image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        combined_features = torch.cat(flat_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, combined_features)
        return inputs_embeds, new_attention_mask, new_labels

    def _forward_with_future(self, model_inputs, labels, forward_kwargs):
        pixel_key = None
        for candidate in ("pixel_values", "pixel_values_videos", "video_values", "video", "videos"):
            if candidate in model_inputs:
                pixel_key = candidate
                break
        if pixel_key is None:
            raise RuntimeError("Future conditioning requires pixel values in the model inputs.")
        pixel_values = model_inputs[pixel_key]
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        image_features = self.backbone.extract_image_tokens(pixel_values, normalize=False)
        num_images = image_features.shape[0]
        if num_images % batch_size != 0:
            raise RuntimeError(
                f"Could not reshape image features by batch: num_images={num_images}, batch={batch_size}"
            )
        frames_per_sample = num_images // batch_size
        context_frame_embeddings = image_features.mean(dim=1).view(batch_size, frames_per_sample, -1)
        predictor_dtype = next(self.future_predictor.parameters()).dtype
        adapter_dtype = next(self.future_adapter.parameters()).dtype
        context_frame_embeddings = context_frame_embeddings.to(dtype=predictor_dtype)
        future_pred = self.future_predictor(context_frame_embeddings, future_frames=self._future_frames)
        future_tokens = self.future_adapter(future_pred.to(dtype=adapter_dtype))
        future_tokens = future_tokens.to(dtype=image_features.dtype)
        current_tokens = image_features.view(batch_size, -1, image_features.shape[-1])
        inputs_embeds, new_attention_mask, new_labels = self._inject_image_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_features=current_tokens,
            extra_features=future_tokens,
        )
        return self.backbone.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **forward_kwargs,
        )

    def _forward_with_belief(self, model_inputs, labels, forward_kwargs):
        pixel_key = None
        for candidate in ("pixel_values", "pixel_values_videos", "video_values", "video", "videos"):
            if candidate in model_inputs:
                pixel_key = candidate
                break
        if pixel_key is None:
            raise RuntimeError("Belief conditioning requires pixel values in the model inputs.")
        pixel_values = model_inputs[pixel_key]
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        image_features = self.backbone.extract_image_tokens(pixel_values, normalize=False)
        num_images = image_features.shape[0]
        if num_images % batch_size != 0:
            raise RuntimeError(
                f"Could not reshape image features by batch: num_images={num_images}, batch={batch_size}"
            )
        frames_per_sample = num_images // batch_size
        context_frame_embeddings = image_features.mean(dim=1).view(batch_size, frames_per_sample, -1)
        belief_dtype = next(self.belief_network.parameters()).dtype
        adapter_dtype = next(self.belief_adapter.parameters()).dtype
        context_frame_embeddings = context_frame_embeddings.to(dtype=belief_dtype)
        state_seq = torch.zeros(
            batch_size,
            frames_per_sample,
            self.belief_network.config.state_dim,
            device=context_frame_embeddings.device,
            dtype=context_frame_embeddings.dtype,
        )
        belief_outputs = self.belief_network(visual_seq=context_frame_embeddings, state_seq=state_seq)
        belief_tokens = self.belief_adapter(belief_outputs["pred_next"].to(dtype=adapter_dtype))
        belief_tokens = belief_tokens.to(dtype=image_features.dtype)
        current_tokens = image_features.view(batch_size, -1, image_features.shape[-1])
        inputs_embeds, new_attention_mask, new_labels = self._inject_image_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_features=current_tokens,
            extra_features=belief_tokens,
        )
        backbone_outputs = self.backbone.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **forward_kwargs,
        )
        return backbone_outputs, belief_outputs

    def forward(self, inputs, labels=None):
        model_inputs = self.backbone._move_inputs_to_device(inputs)
        forward_kwargs = {"return_dict": True}
        if hasattr(self.backbone.model, "config") and hasattr(self.backbone.model.config, "use_cache"):
            forward_kwargs["use_cache"] = False
        labels = labels.to(self.backbone.device) if labels is not None else None
        if self.future_predictor is not None:
            outputs = self._forward_with_future(model_inputs, labels, forward_kwargs)
        elif self.belief_network is not None:
            outputs, belief_outputs = self._forward_with_belief(model_inputs, labels, forward_kwargs)
        else:
            if labels is not None:
                model_inputs["labels"] = labels
            outputs = self.backbone.model(**model_inputs, **forward_kwargs)
            belief_outputs = None
        total_loss = outputs.loss
        result = {"loss": total_loss, "logits": outputs.logits}
        if belief_outputs is not None:
            result["belief_loss"] = belief_outputs["loss"]
            result["belief_recon_loss"] = belief_outputs["recon_loss"]
            result["belief_kl_loss"] = belief_outputs["kl_loss"]
            result["belief_temporal_nce_loss"] = belief_outputs["temporal_nce_loss"]
            if self._train_belief_network and labels is not None and self._belief_aux_weight > 0.0:
                total_loss = total_loss + self._belief_aux_weight * belief_outputs["loss"]
                result["loss"] = total_loss
        return result

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens=64):
        model_inputs = self.backbone._move_inputs_to_device(inputs)
        return self.backbone.model.generate(**model_inputs, max_new_tokens=max_new_tokens)


MultimodalValueModel = MultimodalBeliefModel
