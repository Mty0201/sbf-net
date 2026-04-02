---
phase: 6
slug: semantic-first-route-definition
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | none - documentation and artifact audit |
| **Config file** | none |
| **Quick run command** | `rg -n --glob '!docs/archive/**' --glob '!.planning/milestones/**' "support-only|support-shape|support-guided semantic focus|geometric pressure" docs/canonical train.md .planning/phases/06-semantic-first-route-definition` |
| **Full suite command** | `test -f docs/canonical/sbf_semantic_first_route.md && test -f docs/canonical/sbf_semantic_first_contract.md && rg -n --glob '!docs/archive/**' --glob '!.planning/milestones/**' "support-only is the strongest current reference baseline|support-shape is weaker side evidence only|support-guided semantic focus|no direction target|no side target|no distance target|no ordinal shape pressure" docs/canonical train.md .planning/phases/06-semantic-first-route-definition` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run the quick command above
- **After every plan wave:** Run the full suite command above
- **Before `$gsd-verify-work`:** Confirm the baseline/reference language and candidate-route contract all point to the same support-only-first interpretation
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 1 | MAIN-02, AUX-01 | documentation audit | `rg -n "support-only is the strongest current reference baseline|support-shape is weaker side evidence only|support-guided semantic focus route" docs/canonical/sbf_semantic_first_route.md` | ✅ | ✅ green |
| 6-01-02 | 01 | 1 | MAIN-02, AUX-01 | documentation audit | `rg -n "support-only|support-shape|support-guided semantic focus route|historical/reference evidence" docs/canonical/README.md docs/canonical/sbf_facts.md` | ✅ | ✅ green |
| 6-02-01 | 02 | 1 | MAIN-02, AUX-02 | contract audit | `rg -n "support-guided semantic focus route|no direction target|no side target|no distance target|no ordinal shape pressure|support-only baseline" docs/canonical/sbf_semantic_first_contract.md` | ✅ | ✅ green |
| 6-02-02 | 02 | 1 | AUX-02 | runtime-guidance audit | `rg -n "support-only reference baseline|support-shape side evidence|support-guided semantic focus route|stable runtime entry config" train.md docs/canonical/sbf_training_guardrails.md` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 5s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** passed
