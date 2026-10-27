---
name: liuding-documentation
description: Write or reorganize Liuding Forge README, tutorials, API pages, user guides, and maintainer documentation with the project's user-first, concise documentation policy. Use when documentation is added, updated, translated, indexed, or reviewed in this repository.
---

# Maintain Liuding Forge Documentation

## Choose the reader and location

Write primarily for library users. Use `README.md` and `README.en.md` for the product entry, `docs/tutorials/` for complete learning workflows, `docs/user-guide/` for tasks and troubleshooting, and `docs/api/` for current contracts. Use `CONTRIBUTING.md` or `docs/development/` only for reusable maintainer procedures.

Development reports, audits, historical benchmark results, validation logs, planning history, and raw evidence do not belong in the source repository. Do not expose internal milestone identifiers or naming-policy discussion in user-facing pages.

## Apply the content budget

- Tutorials may include goals, prerequisites, complete steps, runnable code, expected results, recovery, and common errors.
- Other pages keep only current facts, necessary constraints, commands, and support boundaries. Remove background narrative, implementation chronology, repeated status, and speculative catalogs.
- Maintain one source for each fact and link to it. Update an existing page before creating another; merge or remove obsolete pages.
- State implemented, preview, unsupported, and planned capabilities precisely. Do not infer weights, accuracy, device support, or performance from architecture or smoke tests.

Use Chinese for repository documentation at the current stage, except the separate English `README.en.md`. Keep the two README files aligned in structure and support claims. Put each natural-language paragraph or list item on one source line and let the renderer wrap; preserve structural line breaks in code, tables, commands, and trees.

## Verify

Check local links, heading order, table columns, code fences, retired API names, and commands. Compile or run examples whose behavior changed. Keep the document index grouped by user task instead of listing every page, and confirm that no one-time report or local path entered the repository.
