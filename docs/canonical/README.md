# Canonical SBF Guidance Index

This file is the entry point into the canonical SBF guidance set. It exists so maintainers can find repository-specific facts, evidence, and training guardrails without depending on legacy workflow-state files.

## What This Canonical Set Covers

The canonical set answers the minimum repository-specific questions that must stay discoverable during the current milestone:

- SBF versus Pointcept boundary rules and no-fallback expectations
- Current Stage-2 status and the active semantic-first boundary supervision direction
- Experiment evidence and interpretations that still govern future work
- Training entrypoint, config roles, and fail-fast runtime guardrails

The repository still preserves historical `support`, `axis-side`, and `axis + side + support` evidence, but those routes are reference-only and are not the preferred current mainline.

## Four Maintainer Questions

1. Which rules define the SBF versus Pointcept maintenance boundary?
   - Primary answer: `docs/canonical/sbf_facts.md` -> `## SBF vs Pointcept Boundary`
   - Supporting boundary guardrails: `docs/canonical/sbf_facts.md` -> `## Work Boundaries That Still Apply`
2. What is the current Stage-2 state, and how does the semantic-first direction relate to the older geometric-field routes?
   - Stage label and active direction: `docs/canonical/sbf_facts.md` -> `## Current Stage-2 Status`
   - Historical tensor and supervision semantics: `docs/canonical/sbf_facts.md` -> `## Historical Mainline Semantics`
3. Which experiment evidence and conclusions still constrain future work?
   - Confirmed evidence list: `docs/canonical/sbf_facts.md` -> `## Evidence That Still Governs Future Work`
   - Current interpretation of that evidence: `docs/canonical/sbf_facts.md` -> `## Current Interpretation`
4. How should a maintainer launch or review training without creating an invalid run?
   - Required env inputs: `docs/canonical/sbf_training_guardrails.md` -> `## Required Runtime Inputs`
   - Canonical runtime path: `docs/canonical/sbf_training_guardrails.md` -> `## Canonical Training Entrypoint`
   - Config selection and no-relax guardrails: `docs/canonical/sbf_training_guardrails.md` -> `## Config Roles` and `## Guardrails That Must Not Be Relaxed`
   - Explicit reject list for bad runs: `docs/canonical/sbf_training_guardrails.md` -> `## Invalid Run Patterns`

## Read Order

1. Read `docs/canonical/sbf_facts.md` first if you need boundary, stage, active-direction, or evidence answers.
2. Read `docs/canonical/sbf_training_guardrails.md` next if you need runtime entry, config, smoke/full-train, or fail-fast rules.
3. Return to `AGENTS.md` and `docs/workflows/sbf_net_workflow_v1.md` only for workflow boundaries, startup order, and closeout rules; repo-specific facts should be sourced from the canonical docs above.

## Phase 1 Maintainer Self-Check

Use these questions to verify that the canonical docs set still answers the Phase 1 success criteria without opening legacy workflow-state files.

1. Can I identify the SBF-vs-Pointcept boundary and the "stop instead of patching around host issues" rule?
   - Expected answer anchors: `docs/canonical/sbf_facts.md` -> `## SBF vs Pointcept Boundary`; `docs/canonical/sbf_facts.md` -> `## Work Boundaries That Still Apply`
2. Can I confirm the repository is still in `Stage-2 architecture rollout / verification phase`, that semantic-first boundary supervision is the active direction, and that the older geometric-field route is historical/reference evidence?
   - Expected answer anchors: `docs/canonical/sbf_facts.md` -> `## Current Stage-2 Status`; `docs/canonical/sbf_facts.md` -> `## Historical Mainline Semantics`
3. Can I find the experiment scoreboard and the current interpretation that `support-only (reg=1, cover=0.2) = 74.6` remains the best confirmed reference result while `axis-side` full-train is still unverified?
   - Expected answer anchors: `docs/canonical/sbf_facts.md` -> `## Evidence That Still Governs Future Work`; `docs/canonical/sbf_facts.md` -> `## Current Interpretation`
4. Can I recover the only supported training entry, the required env vars, the stable runtime config role, the historical axis-side smoke/full-train commands, and the banned fallback patterns?
   - Expected answer anchors: `docs/canonical/sbf_training_guardrails.md` -> `## Required Runtime Inputs`; `docs/canonical/sbf_training_guardrails.md` -> `## Canonical Training Entrypoint`; `docs/canonical/sbf_training_guardrails.md` -> `## Config Roles`; `docs/canonical/sbf_training_guardrails.md` -> `## Guardrails That Must Not Be Relaxed`; `docs/canonical/sbf_training_guardrails.md` -> `## Invalid Run Patterns`
