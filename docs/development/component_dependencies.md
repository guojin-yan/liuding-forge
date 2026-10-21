# Component Dependency Contract

`component-dependencies.json` is the canonical machine-readable policy for the
eight current entity components. It distinguishes required direct dependencies
from optional allowed adapter edges and records the unique owner of every
public-header and production-source root. Aggregate targets are conveniences;
they cannot satisfy an entity component's required edge.

CMake writes the actual direct target and link-interface edges to
`component-target-graph.tsv` in the build directory. During configuration,
`scripts/validation/component_graph.py` compares that graph with the policy and scans all
`jyppx/...` includes in public headers and production sources. A deterministic
`component-dependency-report.json` is written beside the CMake cache.

The gate rejects undeclared or cyclic edges, missing direct dependencies,
public includes without a matching public link interface, duplicate edges,
unknown ownership roots, and ownership assertion drift. It runs during every
source configuration, including tests-disabled component profiles, and is also
registered as the `ComponentDependencyGraph` CTest for continuous integration.

Any new entity component or dependency edge must update the architecture
decision and the JSON policy in the same change. Do not route a dependency
through `jyppx::all`, `jyppx::sdk`, or another entity target.
