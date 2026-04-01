# SBF Canonical Facts

## Purpose

This document is the canonical, repo-local source for the minimum SBF-specific facts that must survive workflow cleanup. It records the SBF-vs-Pointcept boundary, the current Stage-2 status, the active `axis + side + support` mainline semantics, the work boundaries that still apply, and the experiment evidence that still governs future work.

It is intentionally not a workflow document. A maintainer should be able to recover the core repository facts here without opening `project_memory/`, `handoff/`, or other workflow-state files.

## SBF vs Pointcept Boundary

- Only `semantic-boundary-field` is actively maintained in this repository workflow.
- `Pointcept` is an external host dependency and remains read-only unless the author explicitly authorizes a host-side change.
- If a failure appears to come from the Pointcept host interface, stop and report the boundary issue instead of patching around it inside this repo or inventing a host-side fallback.
- Do not add compatibility layers, silent bypasses, swallowed errors, or other fallback behavior that hides training or integration problems.
- Prefer project-local dataset, model, loss, evaluator, config, and trainer extensions that plug into Pointcept through existing interfaces.
- For longer background on the host boundary, see [docs/pointcept_boundary.md](../pointcept_boundary.md).

## Current Stage-2 Status

- The repository is in `Stage-2 architecture rollout / verification phase`.
- The active mainline is `axis + side + support`.
- The active validation center is `semseg-pt-v3m1-0-base-bf-edge-axis-side-train` plus its smoke config.
- In current author shorthand, `magnitude` means `support`; it does not mean a separate landed branch.
- The current Stage-2 goal is to replace the old explicit signed-direction formulation with the `axis + side + support` expression, not to reopen a broad support-parameter sweep.

## Active Mainline Semantics

- `edge_pred = [axis(3), side_logit(1), support_logit(1)]`.
- The model interface still exposes `seg_logits / edge_pred`, and `edge_pred` remains a 5-channel tensor.
- The current mainline reuses the six-column `edge.npy = [dir_x, dir_y, dir_z, dist, support, valid]`.
- `side` supervision is derived at runtime from the historical direction ground truth; no extra sidecar label file is introduced for the mainline.
- `dist` stays in ground truth for validity-domain checks and historical comparison, but `dist` is not an independent predicted channel on the current mainline.

## Work Boundaries That Still Apply

- Stay inside `semantic-boundary-field`; do not modify code outside this repository without explicit authorization.
- Do not rewrite Pointcept registry, trainer, dataset, or host protocol behavior as part of SBF mainline work.
- Do not overstate validation status: the `axis-side` route is implemented, but implementation alone is not proof of full-train success.
- Prefer exposing real architectural or supervision problems over masking them with compatibility patches.
- Treat historical signed-direction runs as evidence and comparison routes, not as the active mainline semantics.
