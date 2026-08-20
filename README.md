# Belief-VLM: implementation and mathematics

This repository implements video question answering in three increasingly structured modes:

1. **VLM only:** fine-tune InternVL to generate the target answer with supervised causal-language-model loss.
2. **VLM + RL:** use an InternVL hidden state as a state vector and train a separate multiple-choice policy/value network with PPO.
3. **VLM + RL + vector memory:** retrieve similar previous decisions, fuse them with the current VLM state, and then apply PPO.

The main execution sequence is:

```text
raw HD-EPIC annotation + MP4
                |
                v
      resolve clip and sample frames          data_loading.py
                |
                v
      build InternVL video/text inputs        data_loading.py
                |
                v
  Stage 1: InternVL multimodal forward        model.py + train.py
                |
                v
 answer-token logits -> masked cross-entropy
                |
                v
       trained VLM checkpoint
                |
                v
 Stage 2: prompt-only VLM forward -> pooled hidden state
                |
                +------ retrieve past experience ------+
                |                                       |
                v                                       v
          base VLM state <---------------- gated memory fusion
                |
                v
       PPO policy/value network                train_ppo_vqa.py
                |
                v
       choice -> reward -> PPO update
                |
                v
       write experience to memory              vector_memory.py
```

## Files and their place in the pipeline

| File | Main implementation |
|---|---|
| `data_loading.py` | Annotation parsing, video resolution and decoding, deterministic split, task sampling, InternVL prompt construction, padding, and data loaders. |
| `model.py` | InternVL processor/model initialization, device and dtype handling, multimodal forward pass, hidden-state pooling, generation, and auxiliary visual embeddings. |
| `train.py` | Supervised VLM training, AdamW optimization, PEFT, distributed preparation, validation, and VLM checkpoints. |
| `vector_memory.py` | Online experience storage, normalized vector index, nearest-neighbor retrieval, weighted aggregation, and answer-only encoding helpers. |
| `train_ppo_vqa.py` | PPO policy/value model, action masking, reward and advantage construction, clipped PPO loss, memory fusion, optimization, and PPO checkpoints. |

## Notation and tensor shapes

For one example, define

$$x_i=(V_i,q_i,C_i,y_i,d_i,\tau_i),$$

where:

- $V_i$ is a video or temporal video clip;
- $q_i$ is the question;
- $C_i=(c_{i1},\ldots,c_{in_i})$ is the list of answer choices;
- $y_i$ is the correct answer index or answer text;
- $d_i$ is the sample ID;
- $\tau_i$ is the task name, derived from the annotation filename.

The important dimensions are:

| Symbol | Meaning |
|---|---|
| $B$ | batch size |
| $T$ | sampled frames per video, `--video_frames` |
| $L$ | processed text/multimodal sequence length |
| $d$ | InternVL language hidden dimension |
| $A$ | maximum PPO action count, `--max_choice_options` |
| $K$ | number of retrieved memories, `--db_top_k` |

Typical batched values are `input_ids [B,L]`, `attention_mask [B,L]`, labels `[B,L]`, visual inputs such as `pixel_values [B*T,C,H,W]`, VLM hidden states `[B,L,d]`, PPO states `[B,d]`, and policy logits `[B,A]`. Exact visual keys and shapes are processor-dependent.

---

## 1. Raw data and annotation loading

### 1.1 Accepted annotations

`_expand_annotation_paths()` and `_load_single_records()` accept:

- one `.json`, `.jsonl`, or `.csv` file;
- a directory containing those file types;
- a comma-separated list of files.

A JSON file may be a list, contain a list under `data`, `samples`, `annotations`, or `items`, or be a dictionary keyed by sample ID. While merging files, `_load_records()` assigns the filename without its extension as `task_name` unless the record already has one.

Minimum supervised-VLM record:

```json
{
  "id": "sample-001",
  "video_id": "P01-video-01",
  "question": "What is the wearer doing?",
  "answer": "The wearer is opening a drawer."
}
```

Minimum PPO record:

```json
{
  "id": "sample-002",
  "video_id": "P01-video-02",
  "question": "What will the wearer do next?",
  "options": ["Open a drawer", "Wash a cup", "Leave the room"],
  "correct_idx": 2
}
```

In the PPO loader, indices `1...N` are interpreted as one-based and reduced by one; `0` also means the first choice. Consequently, the example above selects `"Wash a cup"`.

### 1.2 Video and clip resolution

`_resolve_hd_epic_video_path()` first tries a direct path in the record. Otherwise it constructs

```text
<video_root>/<participant_id>/<video_id>.<video_extension>
```

and finally tries `<video_root>/<video_id>.<video_extension>`. If `participant_id` is absent, the substring of `video_id` before the first `-` is used.

The clip window is read from nested HD-EPIC `inputs.video*` metadata or top-level `start_time` and `end_time`. Timecodes may be seconds, `MM:SS`, or `HH:MM:SS`.

### 1.3 Frame sampling and decoding

Let the decoded clip cover frame indices $f_s$ through $f_e$, with

$$f_s=\lfloor t_s\,\mathrm{fps}\rfloor,\qquad
f_e=\lfloor t_e\,\mathrm{fps}\rfloor,$$

clipped to valid video bounds. If the clip contains $N=f_e-f_s+1>T$ frames, `_sample_frame_indices()` approximately selects

$$f_j=f_s+\left\lfloor\frac{j(N-1)}{T-1}\right\rfloor,
\qquad j=0,\ldots,T-1.$$

OpenCV seeks directly to these indices so cost normally scales with $T$, rather than with the complete video length. If random seeking fails, the loader scans the clip linearly. If fewer than $T$ images are recovered, the final valid image is repeated until the clip has $T$ frames.

Therefore the visual sample entering the processor is conceptually

$$F_i=[I_{i1},I_{i2},\ldots,I_{iT}].$$

There is no random crop, temporal jitter, or other data augmentation in this loader.

### 1.4 Deterministic train/validation split

For sample ID $d_i$ and seed $s$, `_stable_fold()` computes

$$h_i=\operatorname{int}\left(\operatorname{MD5}(s:d_i)[0:8],16\right),
\qquad u_i=\frac{h_i}{2^{32}-1}.$$

The sample belongs to validation when

$$u_i<r_{\mathrm{val}},$$

where $r_{\mathrm{val}}$ is `--val_ratio`, clipped to $[0,0.5]`. All other samples go to training. Hashing the ID makes the split reproducible and independent of annotation ordering.

### 1.5 Sampling across VQA tasks

`LocalHD_EPICRLVQADataset` groups record indices by `task_name`. In the default `task_uniform` mode, `TaskUniformDistributedSampler` first samples a task uniformly, then a record uniformly inside that task:

$$P(i)=\frac{1}{|\mathcal T|}\frac{1}{N_{\tau_i}},$$

where $\mathcal T$ is the task set and $N_{\tau_i}$ is the number of samples in the selected task. This gives small and large annotation files equal task-level probability. Distributed ranks take strided subsets of the same generated index sequence.

---

## 2. Converting decoded data into InternVL inputs

### 2.1 Processor construction

`build_vlm_processor()` loads `AutoProcessor` and `AutoConfig` for the chosen InternVL checkpoint with remote model code enabled. The processor's image/video resize and crop sizes are overwritten with the model vision configuration's `image_size`, keeping preprocessing consistent with the visual encoder.

### 2.2 Supervised example construction

`build_sft_example()` creates an InternVL chat containing:

```text
user:      video + question + options
assistant: target answer
```

Conceptually, the processor returns a multimodal sequence

$$z_i=[z_i^{\mathrm{prompt}},z_i^{\mathrm{answer}}]$$

together with processed video tensors. Labels initially copy `input_ids`, after which prompt and padding positions are replaced with `-100`:

$$
\ell_{it}=
\begin{cases}
-100, & t\text{ is prompt or padding},\\
z_{it}, & t\text{ is an assistant-answer token}.
\end{cases}
$$

PyTorch/Hugging Face ignores `-100` positions when computing causal-LM loss. The dataset yields:

```text
id, task_name, prompt, answer_text, inputs, labels
```

### 2.3 PPO example construction

`build_prompt_only_example()` applies the same video/question chat template but does not append an assistant answer. `LocalHD_EPICRLVQADataset.__getitem__()` returns:

```text
id, task_name, prompt, choices, correct_idx,
answer_text=choices[correct_idx], inputs
```

The PPO model never generates answer tokens. The prompt is encoded once as a multimodal sequence, and a separate policy chooses an index into `choices`.

### 2.4 Collation

`_stack_inputs()` right-pads `input_ids` and `attention_mask` to the longest sequence in the batch. If each example contains flattened frame pixels `[T,C,H,W]`, it concatenates them into `[B*T,C,H,W]`; other tensors are stacked normally.

`collate_sft_batch()` additionally pads labels with `-100`. `collate_rl_vqa_batch()` retains variable-length Python choice lists and creates:

$$\texttt{correct\_idx}\in\mathbb{N}^{B},\qquad
\texttt{num\_choices}\in\mathbb{N}^{B}.$$

These choice counts later mask invalid policy actions.

---

## 3. InternVL model implementation

### 3.1 Backbone initialization

`InternVLBackbone` in `model.py` performs the following:

1. Map `vl_dtype` to `float16`, `bfloat16`, or `float32`.
2. Load the InternVL config, processor, and tokenizer.
3. Load `AutoModelForImageTextToText`; fall back to `AutoModelForCausalLM` when needed.
4. Optionally apply a quantization configuration.
5. Move the model to the selected device.
6. Freeze all VLM parameters when `freeze_vl=True`.
7. Configure key/value caching through `use_cache`.

Media tensors are moved to the selected floating dtype. Token IDs, masks, and labels keep their integer types.

### 3.2 Multimodal forward computation

Abstractly, InternVL first maps the video frames to visual tokens and combines them with text tokens:

$$E_i^{V}=f_{\mathrm{vision}}(F_i),\qquad
E_i^{T}=f_{\mathrm{embed}}(z_i),$$

$$H_i^{(0)}=\operatorname{Combine}(E_i^{V},E_i^{T}).$$

The language transformer then applies layers

$$H_i^{(\ell+1)}=\operatorname{TransformerLayer}_{\ell}(H_i^{(\ell)}),
\qquad \ell=0,\ldots,L_{\mathrm{layers}}-1.$$

Finally, the language head produces token logits

$$o_{it}=W_{\mathrm{LM}}H_{it}^{(-1)}.$$

`MultimodalVLMModel.forward()` returns at least `loss` and `logits`. With `return_hidden_states=True`, it also returns the selected sequence hidden states and a pooled state.

### 3.3 Hidden-state selection and pooling

`_pool_hidden_states()` accepts `layer_idx`. Negative indices count from the end; out-of-range values are clipped to the available layer interval. Let the selected sequence state be

$$H^{(\ell)}\in\mathbb{R}^{B\times L\times d}.$$

For `pooling="last"`, right-padded sequence length is

$$L_i=\sum_{t=1}^{L}m_{it},$$

and the state is

$$s_i^{(\ell)}=H_{i,L_i-1}^{(\ell)}.$$

For `pooling="mean"`,

$$s_i^{(\ell)}=
\frac{\sum_{t=1}^{L}m_{it}H_{it}^{(\ell)}}
{\max(1,\sum_{t=1}^{L}m_{it})}.$$

The PPO policy uses the final-layer pooled state $s_i^{(-1)}$. Vector-memory queries use $s_i^{(m)}$, where $m$ is `--memory_layer_idx`.

### 3.4 Auxiliary visual-only embedding path

`extract_frame_embeddings()` calls InternVL's `get_image_features()`. If the returned feature for each frame contains $P$ visual tokens, it averages them:

$$e_f=\frac{1}{P}\sum_{p=1}^{P}E_{fp}^{V},\qquad
\hat e_f=\frac{e_f}{\max(\|e_f\|_2,\varepsilon)}.$$

`extract_clip_embeddings()` flattens a list of clips, encodes all frames, splits them back into clips, and stacks them. This is an auxiliary API; the current PPO path queries memory with pooled multimodal language states, not these visual-only embeddings.

---

## 4. VLM-only supervised training

`train.py` implements the first learning stage. For a video-question-answer example, the VLM estimates

$$p_\theta(y_t\mid F_i,q_i,C_i,y_{<t}).$$

Because prompt/padding labels are masked, the implemented answer-token loss is

$$
\mathcal L_{\mathrm{SFT}}(\theta)
=-\frac{1}{N_{\mathrm{answer}}}
\sum_{i=1}^{B}\sum_{t\in\mathcal A_i}
\log p_\theta(y_{it}\mid F_i,q_i,C_i,y_{i,<t}),
$$

where $\mathcal A_i$ contains only assistant-answer positions.

For every batch, `run_epoch()` executes:

```text
inputs, labels = batch
outputs = MultimodalVLMModel(inputs, labels)
loss = outputs["loss"]
backward(loss)
clip gradients
AdamW step
zero gradients
```

The optimizer is AdamW over parameters with `requires_grad=True`:

$$
\theta_{t+1}=\operatorname{AdamW}
(\theta_t,\nabla_\theta\mathcal L_{\mathrm{SFT}},
\eta_{\mathrm{vlm}},\lambda_{\mathrm{wd}}).
$$

The current code has no learning-rate scheduler. Accelerate supplies gradient accumulation, mixed precision, DDP, or FSDP. Training choices are:

- `--peft none`: full fine-tuning unless the VLM was frozen;
- `--peft lora`: freeze base weights and train low-rank adapters;
- `--peft qlora`: quantize base weights to 4-bit NF4 and train LoRA adapters.

A supervised checkpoint stores the VLM state, optimizer state, epoch, global step, arguments, and optional memory. This checkpoint becomes the starting VLM for PPO through `--vlm_checkpoint`.

Minimal Stage-1 command:

```bash
cd Belief-VLM
accelerate launch train.py \
  --dataset_type hd_epic_local \
  --annotation_path /path/to/sft_annotations.json \
  --video_root /path/to/videos \
  --vl_model_preset internvl3_5_2b \
  --video_frames 8 --batch_size 1 \
  --epochs 3 --lr 2e-5 \
  --mixed_precision bf16 \
  --save_dir checkpoints_vlm_sft
```

---

## 5. Online vector memory

The main working memory path is used by PPO. It is **online episodic memory**, not a separately constructed offline database. It begins empty, retrieves only past entries, and is updated after each training batch.

### 5.1 What one memory entry contains

For a selected PPO action $a_i$, memory stores

$$M_i=(\hat k_i,\hat u_i,r_i,d_i,\tau_i,b_i),$$

where:

- $\hat k_i\in\mathbb R^d$ is the normalized intermediate-layer video-question state;
- $\hat u_i\in\mathbb R^d$ is the normalized hidden-state embedding of the selected answer text;
- $r_i$ is the correctness reward;
- $d_i$ and $\tau_i$ are sample and task identifiers;
- $b_i$ is a shortened text such as `Likely goal: Open a drawer.`

The answer embedding is produced by tokenizing

```text
Assistant: <selected choice text>
```

and pooling the VLM at `memory_layer_idx`. The selected policy answer is stored, not the ground-truth answer. Incorrect decisions therefore remain in memory with $r_i=0$.

Before insertion, rows are L2-normalized:

$$\hat k_i=\frac{k_i}{\max(\|k_i\|_2,10^{-12})},\qquad
\hat u_i=\frac{u_i}{\max(\|u_i\|_2,10^{-12})}.$$

`OnlineVectorMemory.add()` appends metadata and NumPy arrays, and inserts the new context keys into the similarity index.

### 5.2 Nearest-neighbor retrieval

For current memory query $q$, first normalize it:

$$\hat q=\frac{q}{\max(\|q\|_2,10^{-12})}.$$

FAISS `IndexFlatIP` or the NumPy backend scores every stored key using

$$c_i=\hat q^\top\hat k_i=\cos(\hat q,\hat k_i).$$

The index over-fetches up to

$$K'=\min(|M|,\max(16K,K+8))$$

candidates. Retrieval excludes entries with the current sample ID. With `same_task_first=True`, same-task neighbors are selected first in similarity order and remaining positions are filled by other tasks.

### 5.3 Aggregating retrieved experience

For selected neighborhood $\mathcal N_K$, similarity weights are

$$
\alpha_i=
\frac{\exp(c_i-c_{\max})}
{\sum_{j\in\mathcal N_K}\exp(c_j-c_{\max})}.
$$

`retrieve_aggregates()` returns

$$
\bar k=\sum_{i\in\mathcal N_K}\alpha_i\hat k_i,
\qquad
\bar u=\sum_{i\in\mathcal N_K}\alpha_i\hat u_i,
$$

$$
\bar r=\sum_{i\in\mathcal N_K}\alpha_i r_i,
\qquad
\bar c=\sum_{i\in\mathcal N_K}\alpha_i c_i,
$$

plus the number of valid retrieved entries. If memory is empty, the code bypasses fusion and uses the base VLM state directly.

### 5.4 Gated fusion with the current VLM state

`GatedMemoryFusion` receives current final-layer state $s$, retrieved context $\bar k$, answer $\bar u$, reward $\bar r$, and similarity $\bar c$. It constructs a feature-wise gate

$$
g=\sigma\left(
\operatorname{MLP}
(\operatorname{LayerNorm}[s;\bar k;\bar u;\bar r;\bar c])
\right),\qquad g\in(0,1)^d.
$$

The memory contribution and final state are

$$m=W_k\bar k+g\odot W_u\bar u+W_r\bar r,$$

$$\tilde s=\operatorname{LayerNorm}(s+m).$$

Only the retrieved answer projection is directly multiplied by the gate. Retrieved similarity affects the gate; retrieved context and reward have their own ungated projections. The PPO policy consumes $\tilde s$.

### 5.5 Why use an intermediate layer as the memory key?

The implementation separates:

$$k=s^{(m)}\quad\text{for stable retrieval},
\qquad s=s^{(-1)}\quad\text{for the task policy}.$$

Earlier/intermediate features can be made more stable while later features remain task-adaptive. With `--freeze_memory_prefix`, language layers `0,...,m-1` are frozen. This reduces movement of the query/key embedding space when PPO also updates the VLM.

### 5.6 Memory persistence and distributed behavior

`state_dict()` saves metadata, context embeddings, answer embeddings, and rewards. Loading reconstructs and normalizes arrays, then rebuilds the selected FAISS/NumPy index.

Memory is process-local under distributed training. Each rank retrieves from its own experience history, and the checkpoint written by the main process contains only that process's memory.

`vector_memory.py` also contains text-prior helpers intended for supervised training: retrieve belief strings, prepend them to prompts, and rebuild the SFT input. In the current file, `augment_prompts` is indented inside `build_answer_only_inputs()` after its return instead of being a method of `OnlineVectorMemory`. Therefore `train.py --use_db_prior` will fail after memory becomes non-empty. PPO memory does not call this method and uses the implemented numeric aggregation path described above.

---

## 6. VLM + PPO implementation

### 6.1 Policy and value network

`PPOAnswerPolicy` maps one $d$-dimensional state through

```text
LayerNorm(d)
Linear(d,d) -> GELU -> Dropout
Linear(d,d) -> GELU
```

to shared representation $z=f_\phi(s)$. Two heads produce

$$l=W_\pi z+b_\pi\in\mathbb R^A,$$

$$V_\phi(s)=W_Vz+b_V\in\mathbb R.$$

For example $i$ with $n_i$ actual choices, `_masked_logits()` sets

$$
\tilde l_{ia}=
\begin{cases}
l_{ia}, & a<n_i,\\
-\infty, & a\ge n_i.
\end{cases}
$$

The categorical policy is

$$\pi_\phi(a\mid s_i)=
\frac{\exp(\tilde l_{ia})}
{\sum_{j=0}^{n_i-1}\exp(\tilde l_{ij})}.$$

### 6.2 Constructing the policy state

`_build_policy_state()` always computes

$$s_i=\operatorname{Pool}(H_i^{(-1)}).$$

It also computes the memory query

$$q_i=\operatorname{Pool}(H_i^{(m)}).$$

If memory is unavailable or empty,

$$\tilde s_i=s_i.$$

Otherwise it retrieves $(\bar k_i,\bar u_i,\bar r_i,\bar c_i)$ and applies `GatedMemoryFusion` to obtain $\tilde s_i$.

In the current implementation, the final-layer state and intermediate-layer query are obtained through two separate VLM forward calls. The query forward is still executed when memory is disabled or empty, although fusion is then bypassed. Each PPO inner epoch repeats these state computations.

### 6.3 Rollout or data-collection phase

For each batch, `run_epoch()` first performs a no-gradient rollout:

1. Encode the video-question batch into $\tilde s_i$.
2. Compute the old action distribution and value.
3. Sample $a_i\sim\pi_{\mathrm{old}}$ during training; use `argmax` during validation.
4. Save $\log\pi_{\mathrm{old}}(a_i\mid\tilde s_i)$.
5. Compare the action with `correct_idx`.

The reward is binary correctness scaled by `--reward_scale`:

$$
r_i=\beta\,\mathbf 1[a_i=y_i],
\qquad \beta=\texttt{reward\_scale}.
$$

Every question is terminal, so there is no next state and no bootstrapping:

$$R_i=r_i.$$

The fixed rollout advantage is

$$A_i=R_i-V_{\mathrm{old}}(\tilde s_i).$$

There is no discount factor, GAE, replay buffer, multi-step trajectory, advantage normalization, or negative reward. This makes the implemented problem a one-step contextual bandit optimized with the PPO clipped objective.

### 6.4 PPO update phase

The same collected batch is re-evaluated `--ppo_epochs` times. At PPO epoch $j$, the implementation recomputes the VLM and memory-fused state and obtains

$$\log\pi_\phi(a_i\mid\tilde s_i),\qquad V_\phi(\tilde s_i).$$

The importance ratio is

$$
\rho_i(\phi)=
\exp\left(
\log\pi_\phi(a_i\mid\tilde s_i)
-\log\pi_{\mathrm{old}}(a_i\mid\tilde s_i)
\right).
$$

With clipping width $\epsilon=$ `--clip_epsilon`, the two policy improvements are

$$U_i=\rho_iA_i,$$

$$C_i=\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i.$$

The policy loss is

$$
\mathcal L_{\mathrm{policy}}
=-\frac{1}{B}\sum_{i=1}^{B}\min(U_i,C_i).
$$

The value target is the immediate return:

$$
\mathcal L_{\mathrm{value}}
=\frac{1}{B}\sum_{i=1}^{B}
(V_\phi(\tilde s_i)-R_i)^2.
$$

Entropy encourages exploration:

$$
\mathcal H(\pi)
=-\frac{1}{B}\sum_i\sum_{a=0}^{n_i-1}
\pi_\phi(a\mid\tilde s_i)
\log\pi_\phi(a\mid\tilde s_i).
$$

The final minimized loss is

$$
\boxed{
\mathcal L_{\mathrm{PPO}}
=\mathcal L_{\mathrm{policy}}
+c_V\mathcal L_{\mathrm{value}}
-c_H\mathcal H(\pi)
}
$$

where $c_V=$ `--value_coef` and $c_H=$ `--entropy_coef`.

### 6.5 Which components receive gradients?

AdamW uses separate parameter groups:

| Component | Default state | Learning rate |
|---|---|---|
| PPO policy/value network | trainable | `policy_lr` |
| Gated memory fusion | trainable when memory is enabled | `policy_lr` |
| InternVL | frozen by default | none |
| InternVL with `--train_vlm_with_rl` | trainable parameters updated | `vlm_lr` |

After backpropagation, gradients are clipped separately for policy, fusion, and—when enabled—the VLM. The optimizer is stepped after every PPO inner pass, subject to Accelerate's accumulation behavior. No scheduler or PPO value clipping is implemented.

### 6.6 Updating memory after PPO

Memory remains fixed during all PPO passes for the current batch. After optimization:

1. Convert each sampled action index back to its choice text.
2. Encode that selected answer at `memory_layer_idx` without gradients.
3. Store the most recently computed query state, selected-answer embedding, correctness reward, sample/task identifiers, and short belief text.

The next training batch can retrieve these new experiences. This creates the causal loop

```text
current state
  -> retrieve only past experiences
  -> choose action
  -> observe correctness reward
  -> update PPO parameters
  -> append current experience
  -> future retrieval
```

### 6.7 Validation

Validation uses greedy actions rather than sampling:

$$a_i=\arg\max_a\pi_\phi(a\mid\tilde s_i).$$

Reported mean reward is

$$\overline r=\beta\times\text{accuracy},$$

and `_evaluate_policy()` separately reports unscaled classification accuracy. Validation reads existing memory but does not add validation examples to it.

---

## 7. Exact end-to-end training order

### Path A: VLM-only baseline

```text
1. Read annotation with question and answer.
2. Resolve video and temporal clip.
3. Uniformly decode T frames.
4. Build InternVL user-video/assistant-answer conversation.
5. Mask prompt and padding labels.
6. Run InternVL forward pass.
7. Compute answer-token causal cross-entropy.
8. Update full VLM or LoRA/QLoRA parameters with AdamW.
9. Save VLM checkpoint.
```

### Path B: VLM + RL baseline

```text
1. Load multiple-choice annotations and sampled video frames.
2. Build prompt-only InternVL input.
3. Load the Stage-1 VLM checkpoint.
4. Pool the final VLM hidden state.
5. Produce categorical action probabilities and value estimate.
6. Sample one answer and compute binary correctness reward.
7. Compute return, advantage, PPO ratio, clipped loss, value loss, entropy.
8. Update policy/value parameters; optionally update InternVL.
9. Validate with greedy answer selection and save checkpoint.
```

### Path C: belief-aware VLM + RL + memory

```text
1. Perform Steps 1-4 of Path B.
2. Pool an intermediate VLM layer as a memory query.
3. Retrieve K similar past video-question states.
4. Softmax-aggregate their states, selected answers, rewards, similarities.
5. Gate and fuse those aggregates with the current final-layer VLM state.
6. Run the PPO policy/value network on the fused state.
7. Optimize the PPO objective.
8. Encode the chosen answer and append the rewarded experience to memory.
9. Save VLM, policy, fusion module, optimizer, and memory together.
```

---

## 8. Running the two training stages

### Stage 1: supervised VLM

```bash
cd Belief-VLM
accelerate launch train.py \
  --dataset_type hd_epic_local \
  --annotation_path /path/to/sft_annotations.json \
  --video_root /path/to/videos \
  --vl_model_preset internvl3_5_2b \
  --video_frames 8 --batch_size 1 \
  --epochs 3 --lr 2e-5 \
  --mixed_precision bf16 \
  --save_dir checkpoints_vlm_sft
```

### Stage 2: frozen VLM + PPO

```bash
accelerate launch train_ppo_vqa.py \
  --dataset_type hd_epic_local \
  --annotation_path /path/to/multiple_choice_annotations \
  --video_root /path/to/videos \
  --vl_model_preset internvl3_5_2b \
  --vlm_checkpoint checkpoints_vlm_sft/ckpt_epoch_2.pt \
  --video_frames 8 --batch_size 4 \
  --epochs 3 --ppo_epochs 4 \
  --clip_epsilon 0.2 --policy_lr 1e-4 \
  --mixed_precision bf16 \
  --save_dir checkpoints_ppo_vqa
```

### Stage 2: VLM + PPO + vector memory

```bash
accelerate launch train_ppo_vqa.py \
  --dataset_type hd_epic_local \
  --annotation_path /path/to/multiple_choice_annotations \
  --video_root /path/to/videos \
  --vl_model_preset internvl3_5_2b \
  --vlm_checkpoint checkpoints_vlm_sft/ckpt_epoch_2.pt \
  --video_frames 8 --batch_size 4 \
  --epochs 3 --ppo_epochs 4 \
  --policy_lr 1e-4 --mixed_precision bf16 \
  --use_db_prior --db_top_k 2 \
  --db_index_backend auto \
  --memory_layer_idx 3 --freeze_memory_prefix \
  --save_dir checkpoints_ppo_memory
```

Add `--train_vlm_with_rl --vlm_lr 2e-5` when PPO should also update the trainable VLM parameters.

## 9. Checkpoints and resume behavior

The PPO checkpoint contains:

```text
model              InternVL/PEFT weights
policy             PPO policy and value weights
memory_fusion      gated-fusion weights or None
optimizer          AdamW state
epoch/global_step  progress metadata
args               serialized command-line configuration
vector_memory      entries and arrays or None
```

`--resume_checkpoint` restores these values. With `--load_model_only`, weights and memory are loaded, while optimizer state and epoch/global step are not resumed.

## 10. Setup and practical constraints

```bash
conda env create -f environment.yml
conda activate llava_video
```

`palmetto_env.yml` is the cluster-oriented alternative and defines environment `ma-vlcm`. Core dependencies are PyTorch, Transformers, Accelerate, OpenCV, Pillow, and NumPy. PEFT/bitsandbytes are needed for LoRA/QLoRA; FAISS and Weights & Biases are optional.

Important implementation constraints:

- PPO is multiple-choice only and supports at most `max_choice_options` actions, default `5`.
- `vl_max_text_len` is passed into the prompt builders, but supervised and prompt-only processor calls currently set `truncation=False`; it does not limit those multimodal sequences.
- The launch `.sh` files contain cluster-specific paths and aggressive experiment defaults; the commands above expose the minimal logical pipeline.
- The memory index grows for the complete run and currently has no eviction, capacity limit, or reward-based filtering.
- A changing VLM can cause stored memory keys to drift relative to new queries; `freeze_memory_prefix` only reduces this problem.
