# Classification metrics

`#include <jyppx/training/metrics/classification.hpp>`

`Precision`, `Recall`, and `F1Score` are fixed-topology classification
metrics. Construction declares the stable metric identity, class count,
aggregation policy, zero-division policy, and optional ignored target label.
Updates accept finite floating logits shaped `{batch, num_classes}` and
integral target labels. Labels are never silently rounded or used to grow the
class topology.

The metrics accumulate an int64 confusion matrix as their only sufficient
statistics. `merge()` therefore supports exact batch-split and rank-split
reduction when descriptors are identical. An empty or all-ignored metric
throws `NotReady`; `average=none` publishes values through
`per_class_scores()` instead of inventing a scalar.

```cpp
using jyppx::training::metrics::ClassificationAverage;
using jyppx::training::metrics::F1Score;

F1Score validation_f1({.metric_id = "metric.validation.f1",
                       .num_classes = 10,
                       .average = ClassificationAverage::macro,
                       .ignore_index = -1});
validation_f1.update(logits, labels);
const double score = validation_f1.compute();
```

Every metric is an exact `CheckpointParticipant`. Its current-only schema
stores canonical immutable topology separately from bounded little-endian
int64 counts. Restore validates the manifest, descriptor bytes, encoded sizes,
and count range before replacing live state. Register different score kinds or
data splits under distinct `metric_id` values.

This migrated API intentionally does not preserve the old lazy class-growth or
floating-label behavior. Callers must declare topology and convert targets to
an integral dtype explicitly.
