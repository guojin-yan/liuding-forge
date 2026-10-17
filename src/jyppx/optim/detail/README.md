<!--
Copyright 2026 JYPPX Training Framework Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
-->

# Optimizer implementation details

This directory contains private implementation fragments shared by concrete
optimizer adapters. Files here are not installed and are not part of the
public API.

`stateful_optimizer_adapter.inc` is instantiated only by the Adam, AdamW, and
RMSprop adapter translation units. Each translation unit supplies its own
public descriptor types, native LibTorch optimizer types, diagnostics, wire
magic, media types, and algorithm family selector before including the
fragment. This keeps checkpoint schemas distinct while centralizing their
common validation and atomic-restore behavior.
