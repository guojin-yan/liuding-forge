# Optional model preview

The `models` component is independently optional and defaults to OFF. Its public target is `jyppx::liuding::models`, with direct `nn` and `interop` dependencies. It has no data loader, training loop, serving endpoint or implicit network access.

| Name | Actual architecture | Default parameters |
| --- | --- | ---: |
| `resnet18` | Basic blocks, 2/2/2/2, 7x7 stem, ImageNet strides | 11,689,512 |
| `resnet50` | Bottlenecks, 3/4/6/3, torchvision v1.5 stride placement | 25,557,032 |
| `mobilenet_v3_small` | 11 inverted residual blocks, depthwise convolutions, SE, hard swish | 2,542,856 |
| `mobilenet_v3_large` | 15 inverted residual blocks, depthwise convolutions, SE, hard swish | 5,483,032 |
| `transformer_encoder` | Six post-norm attention/FFN blocks, width 512, eight heads, FFN 2048, final LayerNorm | 18,915,328 |

All constructors create registered, trainable LibTorch modules. They consume the caller's LibTorch RNG normally and do not change the global seed. For repeatable random initialization, seed before construction and record the LibTorch/device lane. Forward/backward follows normal LibTorch autograd and train/eval behavior; inference callers select eval and an appropriate no-grad guard themselves.

```cpp
#include <jyppx/liuding/models/models.hpp>

auto model = jyppx::liuding::models::resnet18({.num_classes = 10});
auto adapter = jyppx::liuding::models::adapt(model);
adapter.state().eval();
torch::NoGradGuard no_grad;
auto result = adapter.invoke({.positional = {images}, .keyword = {}});
```

Image models accept positive dense floating NCHW inputs on exactly the module's device and dtype and return unnormalized `[N,num_classes]` logits. They do not resize, normalize, decode images or interpret labels. `forward_features` returns the four residual stage maps for ResNet, or the last convolutional feature map for MobileNet. Defaults use three image channels and 1,000 classes. Mobile width multipliers in `[0.25,2]` use divisor-eight rounding. State keys follow the torchvision layouts for these documented schedules; input-channel/classifier or width changes require correspondingly matching weights. BatchNorm training retains LibTorch's minimum sample-size constraints.

The Transformer is an encoder of already embedded floating features, not a tokenizer, BERT, language model or embedding/position-encoding package. Its input and output are `[batch,sequence,model_dimension]`. The explicit public transpose wraps the sequence-first LibTorch C++ encoder. Optional keyword tensors are
`attention_mask` (bool `[S,S]`) and `padding_mask` (bool `[N,S]`); true means a
blocked attention edge or an ignored key. Shape, dtype and device must match.
The caller supplies any causal/position semantics. All-masked attention rows
retain backend behavior and can produce NaNs; the component does not silently
rewrite masks. No token embedding, pooling or classifier is appended. Encoder
options select feature width, heads, FFN width, layer count, ReLU/GELU, dropout
and final normalization; layers use post-norm epsilon `1e-5`.

Each architecture has an owning `ModelAdapter` exposing the same registered
parameters/buffers and exact cardinality/rank/channel/feature schemas. Adapter
creation preserves module identity and mode. Tensor device/dtype and mask
relationship checks run in the concrete model. State mutation, forward/backward
and mode/device changes require caller synchronization. Returned tensors retain
normal storage/autograd semantics; this API does not serialize concurrent calls
or produce independent inference snapshots.

`default_model_registry()` constructs a local catalog with five explicit
factories. `list_models`, `model_info` and `create_model` expose the same entries.
`ModelRegistry::add` supports caller-owned extension and rejects duplicate names
and empty factories. Catalog modification needs caller exclusion; there is no
static initializer or implicit plugin loading. Unknown-model diagnostics include
related built-in family names. `parameter_count` counts registered parameter
elements, excludes buffers and checks overflow. The table describes defaults;
custom options change counts.

Weights are separate artifacts with separate licenses. `available_weights`
records upstream ImageNet candidates as **not provisioned**, with no invented
full SHA-256 or artifact license (`NOASSERTION`). Filename hash prefixes are not
accepted as complete digests. The feature Transformer has no universal
pretrained weights. The component never downloads or deserializes `.pth`/pickle
or accepts executable weights. Upstream code licensing does not establish a
trained artifact's license or a dataset redistribution right.

For approved data-only artifacts, create a `pinned_external` `WeightRecord` with
an exact safetensors `ArtifactRef`, complete SHA-256/size, source and artifact
license provenance. `verify_weights` checks identity and bounded tensor
structure. `load_weights` delegates to the real staged strict/explicit-partial
`interop::import_model_weights` contract and returns its complete mapping report;
it never invents missing state or claims that an artifact name proves topology. Import requires exclusive quiescent state, and callers must discard old storage views and rebuild optimizer state afterward. See the interop weight-import contract for rollback and unsupported topology details.

`cache_weights` accepts an explicit caller transport callback. A cache miss
invokes it once, validates returned bytes, writes private staging and publishes
by a no-replace hard link. A valid hit is reverified without transport. Corrupt
hits fail rather than being silently repaired. Encoded read/response limits
apply; the callback owns transport allocation and timing bounds. The cache root
must remain exclusively controlled by the caller. This is not a hostile-path or
multi-tenant storage sandbox, and a filesystem without hard links fails.
`clean_weight_cache` checks a bounded directory inventory before removal and
only removes old canonical SHA-256 `.safetensors` regular files; other files and symlinks are preserved. Cache users must be quiescent during cleanup.

`tests/unit/models` covers actual models, feature shapes, parameter counts,
gradients, masks, invalid configurations, registry and real strict weight/cache
paths. The optional reference workflow in
`tests/unit/models/reference/generate_reference.py` requires exact observed
Python/PyTorch/torchvision/safetensors versions and an actual Git commit. It uses official architecture implementations with **no pretrained downloads**, exports their actual state plus fixed inputs/outputs to safetensors for all five architectures, then invokes the five compiled C++ comparison tests. The feature encoder case uses a boolean causal mask and a boolean padding mask. The generated fixture uses CPU intra-op threads 2 and inter-op threads 1, and the C++ test process configures the same settings at entry. Exact fixture sizes and full SHA-256 values are fixed independently in the generator and C++ test; a runtime evidence document cannot substitute new unreviewed bytes. Tiny random MobileNet logits are compared after normalizing by the nonzero maximum absolute reference output, with rtol 1e-4 and atol 1e-5, so an all-zero output cannot pass merely because its absolute scale is small. Only actual execution of all five exact named tests, with no skipped or failed XML entries, produces a passed report. An ordinary CTest run without
`JYPPX_LIUDING_MODELS_REFERENCE_DIR` may skip these optional reference cases; that skip
does not satisfy the independent official reference gate.
Hashes, cases, tolerances, source identities and package/device versions remain
with generated evidence. Randomly initialized architecture differentials are
not pretrained quality, official trained-weight licensing or accuracy evidence.

`models_preview_manifest_json()` records this distinction and the pending
independent component gates. Model source, tests, reference workflow and docs must be qualified together on Windows CPU and Ubuntu/CUDA, including the install-only `tests/install/models_consumer`. Reference generation, trained weights and release qualification are not marked successful merely because the corresponding code exists. Missing pretrained artifacts remain an explicit preview qualification item for the release owner.
