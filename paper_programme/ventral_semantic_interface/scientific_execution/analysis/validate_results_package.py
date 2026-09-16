#!/usr/bin/env python3
"""Post-execution output validation (read-only).  Writes validation_report.json and exits
non-zero on any failure.  Run from the worktree root."""
import csv
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from ventral_interface.schema import AR_DIAGNOSTIC_COLUMNS, ITEM_LEVEL_COLUMNS, SCHEMA_VERSION, STRATA  # noqa: E402

PKG = os.path.join(ROOT, "paper_programme", "ventral_semantic_interface", "scientific_execution")
FSD, FIG = os.path.join(PKG, "figure_source_data"), os.path.join(PKG, "figures")
FREEZE = "4ad20048e20da84b9f22a92965098b84c2bf7dd6"
BASE = "79f4e5bd94a9c1f82594f4050028b39c220b82b0"
CONTRACT_SHA = "a5ca7b83717b43ab77cedae0cd3e4bef17901ffff051f5ca90b9340b45bda5d3"
STATES = ["W3_SRC", "W3_REP", "W4_SRC", "W4_REP"]
R = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)


def rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        return next(r), list(r)


def check(name, ok, detail=None):
    R[name] = {"pass": bool(ok), "detail": detail}


# ---- expected files
expected = ["item_level_factorization.tsv", "summary_metrics.json", "ar_diagnostic_native_freear.tsv",
            "DIAGNOSTIC_ONLY_PREFIX_CORRECTION_native_freear.tsv", "VENTRAL_INTERFACE_RESULTS_RECAP.md",
            "logs/00_fresh_preflight.log", "logs/01_execution_environment.txt", "logs/02_execute_attempt1.log",
            "figures/fig1_condition_exact.png", "figures/fig2_rescue_decomposition.png",
            "figures/fig3_c_x_ltm_strata.png", "figures/fig4_ar_first_divergence.png",
            "figures/fig5_source_vs_repaired.png", "figure_source_data/analysis_numbers.json"]
missing = [p for p in expected if not os.path.isfile(os.path.join(PKG, p))]
check("expected_files_exist", not missing, missing)

# ---- item-level TSV
hdr, it = rows(os.path.join(PKG, "item_level_factorization.tsv"))
check("item_level_columns_equal_frozen_schema", hdr == ITEM_LEVEL_COLUMNS)
keys = [(r[hdr.index("state_id")], r[hdr.index("item_index")]) for r in it]
check("item_level_rows_4x29571", len(it) == 4 * 29571, len(it))
check("item_level_state_item_unique", len(set(keys)) == len(keys))
check("item_level_each_state_all_items",
      all(sorted(int(i) for s, i in keys if s == st) == list(range(29571)) for st in STATES))
check("item_level_no_empty_cells", all(all(c != "" for c in r) for r in it))

# ---- AR TSV
ahdr, ar = rows(os.path.join(PKG, "ar_diagnostic_native_freear.tsv"))
check("ar_columns_equal_frozen_schema", ahdr == AR_DIAGNOSTIC_COLUMNS)
akeys = {(r[0], r[1]) for r in ar}
native_fail = {(r[hdr.index("state_id")], r[hdr.index("item_index")]) for r in it
               if r[hdr.index("S0_freear_exact_correct")] == "0"}
check("ar_rows_equal_native_freear_failures", akeys == native_fail and len(akeys) == len(ar), len(ar))
check("ar_prefix_correction_label", all(r[ahdr.index("diag_prefix_correction_label")] ==
                                        "DIAGNOSTIC_ONLY_PREFIX_CORRECTION" for r in ar))
phdr, pc = rows(os.path.join(PKG, "DIAGNOSTIC_ONLY_PREFIX_CORRECTION_native_freear.tsv"))
check("prefix_correction_file_matches_ar", len(pc) == len(ar) and all(
    r[phdr.index("diag_prefix_correction_corrected_exact")] == a[ahdr.index("diag_prefix_correction_corrected_exact")]
    for r, a in zip(pc, ar)))

# ---- summary JSON
S = json.load(open(os.path.join(PKG, "summary_metrics.json")))
check("summary_top_level_keys", sorted(S) == sorted(["schema_version", "contract_sha256", "provenance", "gates",
                                                      "results", "ar_diagnostic_native_freear"]))
check("summary_schema_version", S["schema_version"] == SCHEMA_VERSION)
check("summary_contract_sha", S["contract_sha256"] == CONTRACT_SHA)
check("summary_git_head_is_freeze", S["provenance"]["git_head"] == FREEZE)
check("summary_strata_complete", all(sorted(S["results"][s][v]) == sorted(STRATA)
                                     for s in STATES for v in ("freear", "canonical")))
gate_names = {"R", "A", "B", "D", "C", "SHAT", "H", "E"}
check("all_gates_pass", all(set(S["gates"][s]) == gate_names and all(g["pass"] for g in S["gates"][s].values())
                            for s in STATES))
# transitions sum to denominators; S0 counts recomputed from item level
bad = 0
for s in STATES:
    for v in ("freear", "canonical"):
        for st in STRATA:
            for c in ("S1", "S2", "S3"):
                cell = S["results"][s][v][st][c]
                bad += int(sum(cell["transitions_vs_S0"].values()) != cell["denominator"])
        idx = hdr.index(f"S0_{v}_exact_correct")
        n_ok = sum(int(r[idx]) for r in it if r[hdr.index("state_id")] == s)
        bad += int(n_ok != S["results"][s][v]["ALL_REPETITION_ITEMS"]["S0"]["exact_count"])
check("summary_internal_consistency", bad == 0, bad)

# ---- figure source data reproduce summary / item level
f1h, f1 = rows(os.path.join(FSD, "fig1_condition_exact_all_items.tsv"))
bad = sum(int(int(r[f1h.index("exact_count")]) != S["results"][r[0]][r[1]]["ALL_REPETITION_ITEMS"][r[2]]["exact_count"])
          for r in f1)
check("fig1_source_matches_summary", bad == 0 and len(f1) == 32, bad)
f2h, f2 = rows(os.path.join(FSD, "fig2_native_failure_rescue_decomposition.tsv"))
bad = 0
for r in f2:
    s, v = r[0], r[1]
    for c in ("S1", "S2", "S3"):
        t = S["results"][s][v]["ALL_REPETITION_ITEMS"][c]["transitions_vs_S0"]
        bad += int(int(r[f2h.index(f"{c}_rescues")]) != t["WRONG_TO_CORRECT"])
        bad += int(int(r[f2h.index(f"{c}_regressions")]) != t["CORRECT_TO_WRONG"])
    g = lambda k: int(r[f2h.index(k)])  # noqa: E731
    bad += int(g("S1_and_S3_rescues") + g("S1_only_not_S3") != g("S1_rescues"))
    bad += int(g("S1_and_S3_rescues") + g("S3_only_not_S1") != g("S3_rescues"))
    # stacked panel: S1&S3 + S1-only + (S2 rescues not S1) + none == native failures (S2 rescues all here)
    bad += int(g("S1_and_S3_rescues") + g("S1_only_not_S3") + g("S2_only_not_S1") + g("no_condition_rescues")
               != g("native_failures"))
check("fig2_source_matches_summary_and_partitions", bad == 0, bad)
f3h, f3 = rows(os.path.join(FSD, "fig3_strata_all.tsv"))
bad = sum(int(int(r[f3h.index("exact_count")]) != S["results"][r[0]][r[1]][r[2]][r[3]]["exact_count"]
              or int(r[f3h.index("denominator")]) != S["results"][r[0]][r[1]][r[2]][r[3]]["denominator"]) for r in f3)
check("fig3_source_matches_summary", bad == 0 and len(f3) == 4 * 2 * len(STRATA) * 4, bad)
f4h, f4 = rows(os.path.join(FSD, "fig4_ar_diagnostic_summary.tsv"))
bad = 0
for r in f4:
    a = S["ar_diagnostic_native_freear"][r[0]]
    bad += int(int(r[f4h.index("n_native_freear_failures")]) != a["n_native_freear_failures"])
    bad += int(int(r[f4h.index("divergence_at_eos_position")]) != a["divergence_is_eos_position_count"])
    bad += int(int(r[f4h.index("DIAG_prefix_correction_corrected_exact")]) !=
               a["DIAGNOSTIC_ONLY_PREFIX_CORRECTION"]["corrected_exact_count"])
sh, sc = rows(os.path.join(FSD, "fig4_first_divergence_step_counts.tsv"))
for r in sc:
    bad += int(int(r[2]) != S["ar_diagnostic_native_freear"][r[0]]["divergence_step_counts"][r[1]])
check("fig4_source_matches_summary", bad == 0, bad)
f5h, f5 = rows(os.path.join(FSD, "fig5_source_vs_repaired_conditions.tsv"))
bad = 0
for r in f5:
    w, v, c = r[0], r[1], r[2]
    bad += int(int(r[f5h.index("SRC_exact")]) != S["results"][f"{w}_SRC"][v]["ALL_REPETITION_ITEMS"][c]["exact_count"])
    bad += int(int(r[f5h.index("REP_exact")]) != S["results"][f"{w}_REP"][v]["ALL_REPETITION_ITEMS"][c]["exact_count"])
check("fig5_source_matches_summary", bad == 0, bad)

# ---- inputs, contract, frozen code
cfg = json.load(open(os.path.join(ROOT, "paper_programme/ventral_semantic_interface/ventral_interface_frozen_config.json")))
p = cfg["repository_parent"]
inp = {}
for s in cfg["states"]:
    inp[s["state_id"]] = {
        "source": sha(os.path.join(p, s["source_checkpoint_path"])) == s["source_checkpoint_sha256"],
        "archival": sha(os.path.join(p, s["source_checkpoint_archival_copy"])) == s["source_checkpoint_sha256"],
        "head": (s["repaired_head_path"] is None) or sha(os.path.join(p, s["repaired_head_path"])) == s["repaired_head_sha256"]}
check("input_hashes_unchanged", all(all(v.values()) for v in inp.values()), inp)
check("contract_hash_unchanged",
      sha(os.path.join(ROOT, "paper_programme/ventral_semantic_interface/VENTRAL_INTERFACE_EXPERIMENT_CONTRACT.md")) == CONTRACT_SHA)
frozen = git("ls-tree", "-r", "--name-only", FREEZE).stdout.split()
check("frozen_tracked_tree_byte_identical_to_freeze",
      git("diff", "--quiet", FREEZE, "--", *frozen).returncode == 0
      and git("diff", "--cached", "--quiet", FREEZE, "--", *frozen).returncode == 0)
check("head_descends_from_base", git("merge-base", "--is-ancestor", BASE, "HEAD").returncode == 0)
sums = os.path.join(ROOT, "paper_programme/ventral_semantic_interface/SHA256SUMS")
bad = [l for l in open(sums) if sha(os.path.join(ROOT, l.split("  ", 1)[1].strip())) != l.split()[0]]
check("freeze_SHA256SUMS_verifies", not bad, bad)

out = {"all_pass": all(v["pass"] for v in R.values()), "checks": R}
with open(os.path.join(PKG, "validation_report.json"), "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print(json.dumps({k: v["pass"] for k, v in R.items()}, indent=1))
print("ALL_PASS" if out["all_pass"] else "VALIDATION_FAILED")
sys.exit(0 if out["all_pass"] else 1)
