// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TOOLS_CLI_BUNDLE_SIGNING_HPP
#define JYPPX_TOOLS_CLI_BUNDLE_SIGNING_HPP

#include "cli.hpp"

#include <jyppx/liuding/interop/model_bundle.hpp>

namespace jyppx::liuding::cli {

// The CLI owns no private bytes. The verifier captures its immutable external
// policy; this public-only identity can be rendered without exposing paths.
struct BundleTrust {
  interop::ModelBundleVerifyOptions options;
  std::string key_id;
  std::string policy_sha256;
};

BundleTrust bundle_trust_options(const Arguments& arguments,
                                 interop::ModelBundleLimits limits = {});
Json bundle_trust_document(const BundleTrust& trust);
Json bundle_signing_capabilities();

}  // namespace jyppx::liuding::cli

#endif  // JYPPX_TOOLS_CLI_BUNDLE_SIGNING_HPP
