#!/usr/bin/env python3
"""Post-run validation of the directional-dose package (read-only). Run from the worktree root.
Writes scientific_execution/validation_report_directional_dose.json; exit 1 on any failure."""
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from ventral_directional_dose import ALPHA_KEYS, ALPHAS, CONVENTIONS, STATES  # noqa: E402
from ventral_directional_dose.controls import load_immutable_controls  # noqa: E402
from ventral_directional_dose.schema import ITEM_COLUMNS, STRATA, TRANSITIONS, alpha1_factorization, paired, summarize  # noqa: E402

PROG = os.path.join(ROOT, "paper_programme", "ventral_semantic_directional_dose")
PKG = os.path.join(PROG, "scientific_execution")
FSD = os.path.join(PKG, "figure_source_data")
FREEZE = "ea1098771ac52cf3a4c591ba650e58d73b9836e4"
CONTRACT_SHA = "f5399058d4bf53c313fc4bd179c88b1cebb35348fecf74192a67bb107ec7608c"
RAW_TSV = os.environ.get("DOSE_ITEM_TSV", os.path.join(PKG, "item_level_directional_dose.tsv"))
R = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(name, ok, detail=None):
    R[name] = {"pass": bool(ok), "detail": detail}


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)


expected = ["summary_metrics_directional_dose.json", "validity_gates_directional_dose.json",
            "real_state_geometry_preflight.json", "DIRECTIONAL_DOSE_RESULTS_RECAP.md",
            "logs/00_standalone_preflight.log", "logs/01_execution_environment.txt", "logs/02_execute_attempt1.log",
            "logs/standalone_preflight/preflight_report.json", "logs/standalone_preflight/real_state_geometry_preflight.json",
            "figures/fig1_exact_vs_alpha.png", "figures/fig2_prev_S1_recovery.png", "figures/fig3_transitions_vs_alpha.png",
            "figures/fig4_c_correct_native_wrong.png", "figures/fig5_paired_src_rep.png", "figures/fig6_alpha1_vs_S1_S3.png",
            "figure_source_data/analysis_numbers.json"]
missing = [p for p in expected if not os.path.isfile(os.path.join(PKG, p))]
missing += [] if os.path.isfile(os.path.join(PROG, "CENTRAL_STEERING_HANDOFF_DIRECTIONAL_DOSE_RESULTS.md")) else ["../CENTRAL_STEERING_HANDOFF_DIRECTIONAL_DOSE_RESULTS.md"]
check("expected_files_exist", not missing, missing)
check("item_tsv_available", os.path.isfile(RAW_TSV), RAW_TSV)

# ---- item-level TSV
need = set(["state_id", "item_index", "in_C_population", "C_contract_correct", "dose_case", "shat_norm",
            "retrieved_raw_glove_norm", "a100_s_alpha_norm"] + [f"S{k}_{c}_exact_correct" for k in range(4) for c in CONVENTIONS]
           + [f"prev_S1_rescue_{c}" for c in CONVENTIONS])
for a in ALPHAS:
    for c in CONVENTIONS:
        need |= {f"{ALPHA_KEYS[a]}_{c}_{f}" for f in ("exact_correct", "transition_vs_S0", "prev_S1_rescue", "recovers_prev_S1_rescue")}
rows, keys = [], []
with open(RAW_TSV, newline="", encoding="utf-8") as f:
    rd = csv.reader(f, delimiter="\t")
    hdr = next(rd)
    idx = {k: hdr.index(k) for k in need}
    for r in rd:
        d = {k: r[i] for k, i in idx.items()}
        rows.append(d)
        keys.append((d["state_id"], int(d["item_index"])))
check("item_columns_equal_frozen_schema", hdr == ITEM_COLUMNS)
check("item_rows_4x29571", len(rows) == 4 * 29571, len(rows))
check("item_state_item_unique", len(set(keys)) == len(keys))
check("each_state_complete", all(sorted(i for s, i in keys if s == st) == list(range(29571)) for st in STATES))
alpha_prefixes = sorted({c.split("_")[0] for c in hdr if c[:1] == "a" and c[1:4].isdigit()})
check("exactly_four_alpha_blocks_no_fifth", alpha_prefixes == ["a025", "a050", "a075", "a100"], alpha_prefixes)
check("no_hard_stop_geometry_case", all(r["dose_case"] in ("ORDINARY", "NEAR_COLLINEAR_NLERP") for r in rows),
      sorted({r["dose_case"] for r in rows}))
dev = [abs(float(r["a100_s_alpha_norm"]) - float(r["shat_norm"])) / float(r["shat_norm"]) for r in rows]
differ = sum(1 for r in rows if abs(float(r["shat_norm"]) - float(r["retrieved_raw_glove_norm"])) > 1e-3)
check("alpha1_uses_native_norm_not_S1_vector", max(dev) <= 1e-6 and differ > 0,
      {"max_rel_dev_to_native_norm": max(dev), "items_where_native_and_prototype_norms_differ": differ})

# ---- summary recomputes from item level
S = json.load(open(os.path.join(PKG, "summary_metrics_directional_dose.json")))
rec = summarize(rows)
check("summary_results_recompute_exactly", json.dumps(rec, sort_keys=True) == json.dumps(S["results"], sort_keys=True))
check("paired_recomputes_exactly", json.dumps(paired(rows), sort_keys=True, default=str) == json.dumps(S["paired"], sort_keys=True, default=str))
check("alpha1_factorization_recomputes_exactly",
      json.dumps(alpha1_factorization(rows), sort_keys=True) == json.dumps(S["alpha1_factorization"], sort_keys=True))
bad = 0
for sid in STATES:
    for c in CONVENTIONS:
        for st in STRATA:
            for a in ALPHAS:
                cell = S["results"][sid][c][st][ALPHA_KEYS[a]]
                bad += int(sum(cell[t] for t in TRANSITIONS) != cell["denominator"])
check("transitions_sum_to_denominators", bad == 0, bad)
check("summary_alpha_keys_exactly_four", all(set(k for k in S["results"][sid][c][st] if k != "immutable_controls")
                                             == {"a025", "a050", "a075", "a100"}
                                             for sid in STATES for c in CONVENTIONS for st in STRATA))
cfg = json.load(open(os.path.join(PROG, "ventral_directional_dose_frozen_config.json")))
ic = cfg["immutable_controls"]["item_level"]
ctrl = load_immutable_controls(os.path.join(ROOT, ic["path"]), ic["sha256"])
bad = 0
for sid in STATES:
    for c in CONVENTIONS:
        n_prev = sum(1 for r in ctrl[sid] if r[f"S0_{c}_exact_correct"] == "0" and r[f"S1_{c}_exact_correct"] == "1")
        for a in ALPHAS:
            bad += int(S["results"][sid][c]["PREV_S1_RESCUES"][ALPHA_KEYS[a]]["previous_S1_rescues"] != n_prev)
            bad += int(S["results"][sid][c]["ALL_REPETITION_ITEMS"][ALPHA_KEYS[a]]["previous_S1_rescues"] != n_prev)
        ic_blk = S["results"][sid][c]["ALL_REPETITION_ITEMS"]["immutable_controls"]
        for k in range(4):
            bad += int(ic_blk[f"S{k}_exact_count"] != sum(int(r[f"S{k}_{c}_exact_correct"]) for r in ctrl[sid]))
check("previous_S1_rescues_and_controls_match_immutable_file", bad == 0, bad)

# ---- gates & geometry
G = json.load(open(os.path.join(PKG, "validity_gates_directional_dose.json")))
check("all_validity_gates_pass", G.get("all_pass") is True and G.get("failure") is None
      and all(g["pass"] for sid in STATES for g in G["execution"][sid].values())
      and G["DOSE-H_post"]["pass"])
geo = json.load(open(os.path.join(PKG, "real_state_geometry_preflight.json")))
check("geometry_hard_stop_counts_zero", all(geo[s]["counts"][k] == 0 for s in STATES
                                            for k in ("ZERO_SHAT", "ZERO_PROTOTYPE", "NEAR_ANTIPODAL")))
std = json.load(open(os.path.join(PKG, "logs/standalone_preflight/real_state_geometry_preflight.json")))
check("standalone_and_in_process_geometry_identical", std == geo)

# ---- figure source data
def tsv(name):
    with open(os.path.join(FSD, name), newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))
bad = 0
for r in tsv("fig1_exact_vs_alpha_all_items.tsv"):
    blk = S["results"][r["state_id"]][r["convention"]]["ALL_REPETITION_ITEMS"]
    if r["source"] == "directional_dose":
        bad += int(int(r["exact_count"]) != blk[ALPHA_KEYS[float(r["alpha"])]]["exact_count"])
    else:
        bad += int(int(r["exact_count"]) != blk["immutable_controls"][r["source"].replace("immutable_", "") + "_exact_count"])
for r in tsv("fig2_prev_S1_rescue_recovery.tsv"):
    cell = S["results"][r["state_id"]][r["convention"]]["PREV_S1_RESCUES"][ALPHA_KEYS[float(r["alpha"])]]
    bad += int(int(r["previous_S1_rescues_recovered"]) != cell["previous_S1_rescues_recovered"])
for r in tsv("fig3_transitions_vs_alpha.tsv"):
    cell = S["results"][r["state_id"]][r["convention"]]["ALL_REPETITION_ITEMS"][ALPHA_KEYS[float(r["alpha"])]]
    bad += int(int(r["WRONG_TO_CORRECT"]) != cell["WRONG_TO_CORRECT"]) + int(int(r["CORRECT_TO_WRONG"]) != cell["CORRECT_TO_WRONG"])
for r in tsv("fig4_c_correct_native_wrong_dose.tsv"):
    blk = S["results"][r["state_id"]][r["convention"]]["C_CORRECT_AND_NATIVE_LTM_WRONG"]
    if r["source"] == "directional_dose":
        bad += int(int(r["exact_count"]) != blk[ALPHA_KEYS[float(r["alpha"])]]["exact_count"])
    else:
        bad += int(int(r["exact_count"]) != blk["immutable_controls"][r["source"].replace("immutable_", "") + "_exact_count"])
for r in tsv("fig5_paired_src_rep.tsv"):
    p = S["paired"][r["witness"]][r["convention"]][ALPHA_KEYS[float(r["alpha"])]]
    bad += int(int(r["SRC_exact"]) != p["SRC"]["exact"]) + int(int(r["REP_regressions"]) != p["REP"]["regressions"])
for r in tsv("fig6_alpha1_vs_S1_S3.tsv"):
    x = S["alpha1_factorization"][r["state_id"]][r["convention"]]
    bad += int(int(r["alpha1_rescues"]) != x["alpha1_rescues"]) + int(int(r["alpha1_regressions"]) != x["alpha1_regressions"])
    bad += int(int(r["S1_rescues"]) != x["S1_rescues"]) + int(int(r["S3_regressions"]) != x["S3_regressions"])
check("figure_source_data_match_summary", bad == 0, bad)

# ---- immutability
check("contract_hash_unchanged", sha(os.path.join(ROOT, cfg["contract_path"])) == CONTRACT_SHA)
man = json.load(open(os.path.join(PROG, "IMPLEMENTATION_MANIFEST.json")))
mbad = [p for p, h in man["files_sha256"].items() if sha(os.path.join(ROOT, p)) != h]
check("implementation_manifest_hashes_match", not mbad, mbad)
frozen = git("ls-tree", "-r", "--name-only", FREEZE).stdout.split()
check("frozen_tracked_files_byte_identical_to_freeze_commit",
      git("diff", "--quiet", FREEZE, "--", *frozen).returncode == 0 and git("diff", "--cached", "--quiet", FREEZE, "--", *frozen).returncode == 0)
from scripts.ventral_directional_dose.run_directional_dose import verify_lineage  # noqa: E402
lin = verify_lineage(cfg)
check("closed_S0_S3_outputs_sources_heads_unchanged", lin["pass"], {k: v for k, v in lin.items() if k != "states"})
params = {sid: G["execution"][sid]["DOSE-I_parameters"] for sid in STATES}
check("model_parameters_unchanged_and_equal_closed", all(params[s["state_id"]]["before"] == params[s["state_id"]]["after"]
                                                         == s["closed_params_state_dict_sha256"] for s in cfg["states"]))
for name in ("SHA256SUMS", "SHA256SUMS_CODE"):
    base = PROG if name == "SHA256SUMS" else ROOT
    lines = [l for l in open(os.path.join(PROG, name)).read().splitlines() if l.strip()]
    b = [l for l in lines if sha(os.path.join(base, l.split("  ", 1)[1])) != l.split()[0]]
    check(f"freeze_{name}_verifies", not b, b)
gz = os.path.join(PKG, "item_level_directional_dose.tsv.gz")
if os.path.exists(gz):
    h = hashlib.sha256()
    with gzip.open(gz, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    check("gz_decompresses_to_raw_item_tsv", h.hexdigest() == sha(RAW_TSV), h.hexdigest())

out = {"all_pass": all(v["pass"] for v in R.values()), "checks": R, "raw_item_tsv": RAW_TSV,
       "raw_item_tsv_sha256": sha(RAW_TSV)}
json.dump(out, open(os.path.join(PKG, "validation_report_directional_dose.json"), "w"), indent=1, sort_keys=True, default=str)
print(json.dumps({k: v["pass"] for k, v in R.items()}, indent=1))
print("ALL_PASS" if out["all_pass"] else "VALIDATION_FAILED")
sys.exit(0 if out["all_pass"] else 1)
