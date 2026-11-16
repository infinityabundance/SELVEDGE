#!/usr/bin/env python3
"""Append the S30 (§18) number blocks to docs/perf-numbers.json (one-off helper).

Each block is (id, receipt, pointer, format). The helper refuses to duplicate an
id and writes the manifest back with the same structure and sort order the
existing file uses. It is idempotent: re-running replaces the S30 blocks with the
same definitions.
"""

from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "docs", "perf-numbers.json")
DOC = "docs/CAPABILITY-COURT.md"

FORENSICS = "results/capability/forensics/forensics-20261116T173500Z.json"
LADDER = "results/capability/ladder/D-20261116T173600Z.json"
CALIBRATION_ANALYSIS = "results/capability/calibration-analysis-20261116T165500Z.json"
PILOT = "results/capability/pilot/pilot-analysis-20261116T173200Z.json"
COURT_D_SELECT = "results/courts/oracle-20261116T164814Z-d8834a.json"
COURT_X_SELECT = "results/courts/oracle-20261116T164924Z-272401.json"
COURT_LADDER = "results/courts/oracle-20261116T165324Z-55a1d3.json"
FORENSICS_STAMP, FORENSICS_REV = "20261116T165158Z", "dddbc8e0bc58a2047bd1877d5676c5af34d6280e"
LADDER_STAMP, LADDER_REV = "20261116T165340Z", "2a12b749cdebcbc27fbc886643222e3e979c9cf0"
PILOT_STAMP, PILOT_REV = "20261116T165202Z", "dddbc8e0bc58a2047bd1877d5676c5af34d6280e"

BLOCKS = [
    # --- prompt forensics
    ("s30-prompt-max-tokens", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/adapter_truncation/max_prompt_tokens", "{value} tokens"),
    ("s30-prompt-truncated-items", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/adapter_truncation/truncated_items", "{value} items truncated"),
    ("s30-prompt-example-patch", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/prompts/0/example_letter", "{value}"),
    ("s30-prompt-example-diagnostic", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/prompts/3/example_letter", "{value}"),
    ("s30-d-patch-items", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/collapse/patch/items", "{value} patch items"),
    ("s30-d-patch-letters", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/collapse/patch/answers_with_a_letter", "{value} recoverable letters"),
    ("s30-d-patch-constant", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/collapse/patch/constant_letter", "{value}"),
    ("s30-d-diag-items", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/collapse/diagnostic/items", "{value} diagnostic items"),
    ("s30-d-diag-letters", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/collapse/diagnostic/answers_with_a_letter", "{value} recoverable letters"),
    ("s30-d-diag-constant", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/collapse/diagnostic/constant_letter", "{value}"),
    ("s30-x-patch-letters", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/X/collapse/patch/answers_with_a_letter", "{value} recoverable letters"),
    ("s30-x-patch-items", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/X/collapse/patch/items", "{value} patch items"),
    ("s30-x-diag-letters", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/X/collapse/diagnostic/answers_with_a_letter", "{value} recoverable letters"),
    ("s30-x-diag-items", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/X/collapse/diagnostic/items", "{value} diagnostic items"),
    ("s30-collapse-verdict", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/diagnosis/verdict", "{value}"),
    ("s30-d-body-entropy", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/screens/body-completion/entropy_bits", "{value:.3f} bits"),
    ("s30-d-body-screen", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/screens/body-completion/screen", "{value}"),
    ("s30-d-file-entropy", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/screens/file-replacement/entropy_bits", "{value:.3f} bits"),
    ("s30-d-file-screen", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/screens/file-replacement/screen", "{value}"),
    ("s30-d-patch-entropy-before", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/screens/patch-selection/entropy_bits", "{value:.3f} bits"),
    ("s30-d-patch-screen-before", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/screens/patch-selection/screen", "{value}"),

    ("s30-d-body-modal-share", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/screens/body-completion/modal_share", "{value:.3f}"),
    ("s30-d-patch-modal-share", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/screens/patch-selection/modal_share", "{value:.3f}"),
    ("s30-d-file-modal-share", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/screens/file-replacement/modal_share", "{value:.3f}"),
    ("s30-d-file-passes-s29", CALIBRATION_ANALYSIS, None, None,
     "/per_format/file-replacement/by_arm/D/executable_passes", "{value}"),
    ("s30-d-file-items-s29", CALIBRATION_ANALYSIS, None, None,
     "/per_format/file-replacement/by_arm/D/executable_items", "{value}"),

    # --- the work log (same receipts, different document)
    ("worklog-s30-d-patch-letters", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/collapse/patch/answers_with_a_letter", "{value} recoverable letters"),
    ("worklog-s30-d-patch-items", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/collapse/patch/items", "{value} patch items"),
    ("worklog-s30-d-patch-constant", FORENSICS, FORENSICS_STAMP, FORENSICS_REV,
     "/arms/D/collapse/patch/constant_letter", "{value}"),
    ("worklog-s30-mcq-correct", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/mcq_plain_correct", "{value}"),
    ("worklog-s30-mcq-items", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/mcq_plain_items", "{value}"),
    ("worklog-s30-content-stable", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/patch_items_content_stable", "{value}"),
    ("worklog-s30-ladder-items", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/patch_items", "{value}"),
    ("worklog-s30-exec-items", PILOT, PILOT_STAMP, PILOT_REV,
     "/item_outcomes/executable_items", "{value} executable items"),
    ("worklog-s30-both", PILOT, PILOT_STAMP, PILOT_REV,
     "/item_outcomes/both_pass", "{value} are passed by both arms"),
    ("worklog-s30-neither", PILOT, PILOT_STAMP, PILOT_REV,
     "/item_outcomes/neither_pass", "{value} by neither"),
    ("worklog-s30-discordant", PILOT, PILOT_STAMP, PILOT_REV,
     "/item_outcomes/discordant_pairs", "{value} by exactly one"),
    ("worklog-s30-d-patch-passes", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/per_format/patch-selection/executable_passes", "{value}"),
    ("worklog-s30-patch-items", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/per_format/patch-selection/executable_items", "{value}"),
    ("worklog-s30-baseline", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/baseline", "{value:.6f}"),
    ("worklog-s30-variance", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/variance_of_paired_difference", "{value:.6f}"),
    ("worklog-s30-required-n", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/required_n/n_per_category_ceil", "{value} items per category"),
    ("worklog-s30-available", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/n_available", "{value} available per category"),
    ("worklog-s30-mde", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/minimum_detectable_effect/absolute_effect", "{value:.4f} absolute"),

    # --- the sanity ladder on D
    ("s30-ladder-conditions", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/conditions", "{value} conditions"),
    ("s30-ladder-mcq-correct", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/mcq_plain_correct", "{value}"),
    ("s30-ladder-mcq-items", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/mcq_plain_items", "{value}"),
    ("s30-ladder-example-wrong-letter", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/mcq_example_conditions/mcq_example_wrong/letter", "{value}"),
    ("s30-ladder-example-wrong-correct", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/mcq_example_conditions/mcq_example_wrong/correct_letter", "{value}"),
    ("s30-ladder-patch-items", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/patch_items", "{value} items"),
    ("s30-ladder-content-stable", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/patch_items_content_stable", "{value}"),
    ("s30-ladder-controls-copied", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/controls_that_equal_the_example", "{value}"),
    ("s30-ladder-interpretation", LADDER, LADDER_STAMP, LADDER_REV,
     "/verdict/interpretation", "{value}"),
    ("s30-ladder-peak", LADDER, LADDER_STAMP, LADDER_REV,
     "/resources/process_maxrss_bytes", "{value} bytes"),
    ("s30-ladder-vram", LADDER, LADDER_STAMP, LADDER_REV,
     "/resources/vram/max_memory_allocated_gib", "{value:.3f} GiB"),

    # --- the pilot
    ("s30-pilot-selection-items", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/ranking_agreement/content-mean/items", "{value} selection items"),
    ("s30-pilot-d-patch-passes", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/per_format/patch-selection/executable_passes", "{value}"),
    ("s30-pilot-d-patch-items", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/per_format/patch-selection/executable_items", "{value}"),
    ("s30-pilot-x-patch-passes", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/X/per_format/patch-selection/executable_passes", "{value}"),
    ("s30-pilot-x-patch-items", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/X/per_format/patch-selection/executable_items", "{value}"),
    ("s30-pilot-d-diag-passes", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/per_format/diagnostic-identification/diagnostic_passes", "{value}"),
    ("s30-pilot-d-diag-items", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/per_format/diagnostic-identification/diagnostic_items", "{value}"),
    ("s30-pilot-x-diag-passes", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/X/per_format/diagnostic-identification/diagnostic_passes", "{value}"),
    ("s30-pilot-x-diag-items", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/X/per_format/diagnostic-identification/diagnostic_items", "{value}"),
    ("s30-pilot-exec-items", PILOT, PILOT_STAMP, PILOT_REV,
     "/item_outcomes/executable_items", "{value} executable items"),
    ("s30-pilot-both", PILOT, PILOT_STAMP, PILOT_REV,
     "/item_outcomes/both_pass", "{value} passed by both arms"),
    ("s30-pilot-neither", PILOT, PILOT_STAMP, PILOT_REV,
     "/item_outcomes/neither_pass", "{value} passed by neither"),
    ("s30-pilot-discordant", PILOT, PILOT_STAMP, PILOT_REV,
     "/item_outcomes/discordant_pairs", "{value} by exactly one"),
    ("s30-pilot-sign-p", PILOT, PILOT_STAMP, PILOT_REV,
     "/item_outcomes/sign_test_p", "{value:.3f}"),
    ("s30-pilot-baseline", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/baseline", "{value:.6f}"),
    ("s30-pilot-variance", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/variance_of_paired_difference", "{value:.6f}"),
    ("s30-pilot-categories", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/categories", "{value} categories"),
    ("s30-pilot-required-n", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/required_n/n_per_category_ceil", "{value} items per category"),
    ("s30-pilot-n-available", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/n_available", "{value} available per category"),
    ("s30-pilot-mde", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/minimum_detectable_effect/absolute_effect", "{value:.4f} absolute"),
    ("s30-pilot-mde-relative", PILOT, PILOT_STAMP, PILOT_REV,
     "/power/minimum_detectable_effect/relative_to_baseline", "{value:.2f}x the baseline"),
    ("s30-pilot-screen-min-bits", PILOT, PILOT_STAMP, PILOT_REV,
     "/entropy_screen/min_answer_entropy_bits", "{value} bits"),
    ("s30-pilot-screen-max-share", PILOT, PILOT_STAMP, PILOT_REV,
     "/entropy_screen/max_modal_answer_share", "{value}"),
    ("s30-pilot-d-patch-entropy", PILOT, PILOT_STAMP, PILOT_REV,
     "/entropy_screen/by_arm/D/patch-selection/entropy_bits", "{value:.3f} bits"),
    ("s30-pilot-d-diag-entropy", PILOT, PILOT_STAMP, PILOT_REV,
     "/entropy_screen/by_arm/D/diagnostic-identification/entropy_bits", "{value:.3f} bits"),
    ("s30-pilot-x-patch-entropy", PILOT, PILOT_STAMP, PILOT_REV,
     "/entropy_screen/by_arm/X/patch-selection/entropy_bits", "{value:.3f} bits"),
    ("s30-pilot-x-diag-entropy", PILOT, PILOT_STAMP, PILOT_REV,
     "/entropy_screen/by_arm/X/diagnostic-identification/entropy_bits", "{value:.3f} bits"),
    ("s30-pilot-d-margin-patch", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/margins/patch-selection/median", "{value:.4f} nats"),
    ("s30-pilot-d-margin-diag", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/margins/diagnostic-identification/median", "{value:.4f} nats"),
    ("s30-pilot-rank-content-mean", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/ranking_agreement/content-mean/accuracy", "{value:.4f}"),
    ("s30-pilot-rank-content-sum", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/ranking_agreement/content-sum/accuracy", "{value:.4f}"),
    ("s30-pilot-rank-letter-mean", PILOT, PILOT_STAMP, PILOT_REV,
     "/per_arm/D/ranking_agreement/letter-mean/accuracy", "{value:.4f}"),
    ("s30-pilot-spread-patch", PILOT, PILOT_STAMP, PILOT_REV,
     "/cells/patch-selection/accuracy_spread", "{value:.4f}"),
    ("s30-pilot-spread-diag", PILOT, PILOT_STAMP, PILOT_REV,
     "/cells/diagnostic-identification/accuracy_spread", "{value:.4f}"),
    ("s30-pilot-d-peak", PILOT, PILOT_STAMP, PILOT_REV,
     "/resources/D/process_maxrss_gib", "{value:.3f} GiB process peak"),
    ("s30-pilot-d-cgroup", PILOT, PILOT_STAMP, PILOT_REV,
     "/resources/D/cgroup_memory_peak_gib", "{value:.3f} GiB cgroup peak"),
    ("s30-pilot-d-vram", PILOT, PILOT_STAMP, PILOT_REV,
     "/resources/D/vram_max_memory_allocated_gib", "{value:.3f} GiB VRAM allocated"),
    ("s30-pilot-x-peak", PILOT, PILOT_STAMP, PILOT_REV,
     "/resources/X/process_maxrss_gib", "{value:.3f} GiB process peak"),
    ("s30-pilot-x-cgroup", PILOT, PILOT_STAMP, PILOT_REV,
     "/resources/X/cgroup_memory_peak_gib", "{value:.3f} GiB cgroup peak"),
    ("s30-pilot-x-vram", PILOT, PILOT_STAMP, PILOT_REV,
     "/resources/X/vram_max_memory_allocated_gib", "{value:.3f} GiB VRAM allocated"),
    ("s30-pilot-d-limit", COURT_D_SELECT, None, None, "/limits/memory", "{value}"),
    ("s30-pilot-x-limit", COURT_X_SELECT, None, None, "/limits/memory", "{value}"),
    ("s30-ladder-limit", COURT_LADDER, None, None, "/limits/memory", "{value}"),
]


def main() -> int:
    with open(MANIFEST, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    original = list(manifest["blocks"])
    existing = {block["id"]: block for block in manifest["blocks"]}
    defined = {}
    added = 0
    for block_id, receipt, stamp, revision, pointer, fmt in BLOCKS:
        doc = "docs/WORK-LOG.md" if block_id.startswith("worklog-s30-") else DOC
        if stamp is None or revision is None:
            ## Derive the declaration from the receipt itself, so a transcription
            ## can never disagree with the evidence it points at.
            with open(os.path.join(REPO, receipt), "r", encoding="utf-8") as handle:
                document = json.load(handle)
            stamp, revision = document["stamp"], document["revision"]
        block = {
            "declared_receipt_stamp": stamp,
            "declared_revision": revision,
            "doc": doc,
            "format": fmt,
            "id": block_id,
            "sentinel": block_id,
            "source": {"pointer": pointer, "receipt": receipt},
        }
        defined[block_id] = block
        if existing.get(block_id) != block:
            added += 1
    manifest["blocks"] = []
    seen = set()
    for block in original:
        if block["id"] in defined:
            manifest["blocks"].append(defined[block["id"]])
        else:
            manifest["blocks"].append(block)
        seen.add(block["id"])
    for block_id, block in defined.items():
        if block_id not in seen:
            manifest["blocks"].append(block)
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"appended or updated {added} S30 blocks; {len(manifest['blocks'])} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
