<!--
Copyright 2026 JYPPX Training Framework Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
-->

# Optimizer implementation details

This directory contains private implementation fragments shared by concrete
optimizer adapters. Files here are not installed and are not part of the
public API.

`adam_family_adapter.inc` is instantiated only by the Adam and AdamW adapter
translation units. Each translation unit supplies its own public descriptor
types, native LibTorch optimizer types, diagnostics, wire magic, and media
types before including the fragment. This keeps checkpoint schemas distinct
while centralizing their common validation and atomic-restore behavior.
