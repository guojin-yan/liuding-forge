# Local serving Preview host

The opt-in `jyppx_serving_jsonl` example uses the official nlohmann JSON parser,
a fixed application-owned two-feature native model, a fully verified safetensors
bundle, a known-vector warmup, the bounded scheduler and the unary service adapter.
It reads stdin and writes stdout. It opens no network listener.

```text
jyppx_serving_jsonl --demo
{"request_id":"example-1","input":[1,2]}
{"request_id":"example-2","input":[3,4],"deadline_ms":5000}
```

Outputs contain status `ok` and `output:[2,-6]`, then `output:[6,-12]`.
`--demo --health` reports actual readiness; `--manifest` prints the capability
manifest without loading a model. `--cuda 0` selects that exact available device.
`--bundle PATH` accepts an already published compatible local example bundle.
Bundle paths are process configuration, never request fields.

Lines are capped at 4 KiB before parsing; depth is capped at eight; duplicate and
unknown members fail. Input is exactly two finite numbers in [-1e6,1e6]. Deadlines
are cooperative 0..60000 ms. Authentication is the local operating-system
operator and fixed local project; remote authentication/TLS is outside this
example. The host processes JSONL lines sequentially, so dynamic batching is
exercised by the concurrent unit/installation consumers, not claimed by this
sequential host. No pretrained weights or external model downloads are used.
