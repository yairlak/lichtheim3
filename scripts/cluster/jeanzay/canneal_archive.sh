#!/bin/bash
# ===========================================================================
# CANNEAL u1400 -> u2000 -- immutable STORE archive.
#
# Freezes the completed anneal block following the layout of the u850 / u1200
# / final-report archives: one directory per run holding metrics.tsv,
# config.json, provenance.json, logs/, the selected checkpoints and any
# error_audit_u*/ produced for it, each run carrying its own SHA256SUMS with
# ./-relative paths, plus _canneal_report/, _slurm_logs/<jobid>/ and a
# top-level SHA256SUMS over the whole tree.
#
# WHAT IS ARCHIVED
#   * final_canneal_ctrl_h512_s{19..22}  endpoint step_05556000.pt  (the state
#     everything downstream continues from: C errors [52,62,51,56], mean 55.25)
#   * final_canneal_5e5_h512_s{19..22}   endpoint step_05556000.pt
#   * final_canneal_3e5_h512_s{19..22}   endpoint step_05556000.pt
#   * all 12 runs: metrics.tsv, config.json, provenance.json, logs/,
#     error_audit_u*/ if present
#   * _lineage_u1400/: the four final_rep_rescue123_h512_s<seed>
#     step_03889200.pt branch points.  The anneal cannot be reproduced from
#     its own endpoints alone -- it starts there -- so the branch point is
#     part of "required to reproduce all CANNEAL trajectories".  Set
#     L3_INCLUDE_SOURCE=0 to omit them (they stay on SCRATCH either way).
#   * _canneal_report/ in full
#   * ARCHIVE_MANIFEST.json: commit, driver blob, GloVe/lexicon/C/N hashes,
#     per-run endpoint hashes, the u2000 result, and the file/byte counts.
#
# NOTHING IS EVER DELETED FROM SCRATCH.  This script only reads SCRATCH.
#
# VERIFICATION.  Every archived file is hashed at the source AND at the
# destination and the two are compared.  CANNEAL_STORE_VERIFIED=YES is printed
# only if every single pair matches and no file is missing on either side.
# The tree is then made read-only (a-w) so the archive is immutable.
#
# Runs on a login node or, for a tidier accounting, under
#   srun --account=llg@v100 --partition=prepost --time=01:00:00 --pty bash
# The payload is ~500 MB, so either is fine.
#
# Usage:
#   export L3_REPO=<pinned worktree or the main checkout>
#   export L3_RUNS=/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs
#   export L3_STORE=/lustre/fsstor/projects/rech/llg/uss35bp
#   bash scripts/cluster/jeanzay/canneal_archive.sh
# ===========================================================================
set -euo pipefail

STORE=${L3_STORE:-/lustre/fsstor/projects/rech/llg/uss35bp}
RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
REPO=${L3_REPO:-/lustre/fswork/projects/rech/llg/uss35bp/lichtheim3/lichtheim3}
INCLUDE_SOURCE=${L3_INCLUDE_SOURCE:-1}

ARCHIVE_NAME=${L3_ARCHIVE_NAME:-canneal_u2000}
DEST="$STORE/lichtheim3_archives/$ARCHIVE_NAME"

SRC_STEP=3889200            # u1400 branch point
END_STEP=5556000            # u2000 endpoint
ENDPOINT=step_$(printf '%08d' $END_STEP).pt
SOURCE_CKPT=step_$(printf '%08d' $SRC_STEP).pt

GLOVE=${L3_GLOVE:-$REPO/data/glove.6B.300d.txt}
GLOVE_SHA_EXPECTED=91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed
TRAIN_BLOB_EXPECTED=95295d63560ae4c235a6beee8dfb47166f4ed30d
C_POP_SHA=10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50
N_POP_SHA=78e46871d86efaaf34e9ef41891eb22afa93cc58467e4262225b79e8caffd50f

CTRL=(final_canneal_ctrl_h512_s19 final_canneal_ctrl_h512_s20 \
      final_canneal_ctrl_h512_s21 final_canneal_ctrl_h512_s22)
MID=(final_canneal_5e5_h512_s19 final_canneal_5e5_h512_s20 \
     final_canneal_5e5_h512_s21 final_canneal_5e5_h512_s22)
LOW=(final_canneal_3e5_h512_s19 final_canneal_3e5_h512_s20 \
     final_canneal_3e5_h512_s21 final_canneal_3e5_h512_s22)
ALL_RUNS=("${CTRL[@]}" "${MID[@]}" "${LOW[@]}")
LINEAGE=(final_rep_rescue123_h512_s19 final_rep_rescue123_h512_s20 \
         final_rep_rescue123_h512_s21 final_rep_rescue123_h512_s22)

echo "=========================================================="
echo "CANNEAL u1400 -> u2000  STORE ARCHIVE"
echo "  source (read only) : $RUNS"
echo "  destination        : $DEST"
echo "  include u1400 srcs : $INCLUDE_SOURCE"
echo "=========================================================="

# ---- immutability: never write into an existing archive -------------------
if [[ -e "$DEST" ]]; then
    echo "FATAL: $DEST already exists."
    echo "       This archive is immutable by construction.  To create a"
    echo "       revision, export L3_ARCHIVE_NAME=${ARCHIVE_NAME}_v2."
    exit 1
fi

# ---- repo provenance ------------------------------------------------------
cd "$REPO"
COMMIT=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD || echo detached)
BLOB=$(git rev-parse HEAD:scripts/naming_comprehension/train_joint_scratch.py)
[[ "$BLOB" == "$TRAIN_BLOB_EXPECTED" ]] || {
    echo "FATAL: driver blob $BLOB != $TRAIN_BLOB_EXPECTED"; exit 1; }
echo "== repo $REPO @ $COMMIT ($BRANCH)"
echo "== driver blob verified: $BLOB"

[[ -f "$GLOVE" ]] || { echo "FATAL: GloVe not found at $GLOVE"; exit 1; }
GLOVE_SHA=$(sha256sum "$GLOVE" | cut -d' ' -f1)
[[ "$GLOVE_SHA" == "$GLOVE_SHA_EXPECTED" ]] || {
    echo "FATAL: GloVe sha256 $GLOVE_SHA != $GLOVE_SHA_EXPECTED"; exit 1; }
echo "== GloVe verified: $GLOVE_SHA"

# ---- everything must exist BEFORE anything is copied ----------------------
MISSING=0
for R in "${ALL_RUNS[@]}"; do
    for F in metrics.tsv config.json provenance.json \
             "checkpoints/$ENDPOINT"; do
        [[ -e "$RUNS/$R/$F" ]] || { echo "MISSING: $RUNS/$R/$F"; MISSING=1; }
    done
done
if (( INCLUDE_SOURCE )); then
    for R in "${LINEAGE[@]}"; do
        [[ -e "$RUNS/$R/checkpoints/$SOURCE_CKPT" ]] \
            || { echo "MISSING: $RUNS/$R/checkpoints/$SOURCE_CKPT"; MISSING=1; }
    done
fi
REPORT_DIR="$RUNS/_canneal_report"
[[ -d "$REPORT_DIR" ]] || { echo "MISSING: $REPORT_DIR"; MISSING=1; }
(( MISSING == 0 )) || { echo "FATAL: refusing to archive an incomplete set"; exit 1; }
echo "== all expected sources present"

mkdir -p "$DEST"

# ---- copy ----------------------------------------------------------------
copy_run () {                      # $1 = run id
    local R="$1" S="$RUNS/$1" D="$DEST/$1"
    mkdir -p "$D/checkpoints"
    cp -p "$S/metrics.tsv" "$S/config.json" "$S/provenance.json" "$D/"
    cp -p "$S/checkpoints/$ENDPOINT" "$D/checkpoints/"
    [[ -d "$S/logs" ]] && cp -pR "$S/logs" "$D/"
    for A in "$S"/error_audit_u*; do
        [[ -d "$A" ]] && cp -pR "$A" "$D/"
    done
    ( cd "$D" && find . -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum > SHA256SUMS )
    echo "   archived $R  ($(find "$D" -type f | wc -l | tr -d ' ') files)"
}

echo "== copying the 12 anneal runs"
for R in "${ALL_RUNS[@]}"; do copy_run "$R"; done

if (( INCLUDE_SOURCE )); then
    echo "== copying the four u1400 branch points"
    mkdir -p "$DEST/_lineage_u1400"
    for R in "${LINEAGE[@]}"; do
        mkdir -p "$DEST/_lineage_u1400/$R/checkpoints"
        cp -p "$RUNS/$R/checkpoints/$SOURCE_CKPT" \
              "$DEST/_lineage_u1400/$R/checkpoints/"
        [[ -f "$RUNS/$R/metrics.tsv" ]] \
            && cp -p "$RUNS/$R/metrics.tsv" "$DEST/_lineage_u1400/$R/"
        [[ -f "$RUNS/$R/config.json" ]] \
            && cp -p "$RUNS/$R/config.json" "$DEST/_lineage_u1400/$R/"
        [[ -f "$RUNS/$R/provenance.json" ]] \
            && cp -p "$RUNS/$R/provenance.json" "$DEST/_lineage_u1400/$R/"
        for A in "$RUNS/$R"/error_audit_u1400; do
            [[ -d "$A" ]] && cp -pR "$A" "$DEST/_lineage_u1400/$R/"
        done
    done
    ( cd "$DEST/_lineage_u1400" && find . -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum > SHA256SUMS )
fi

echo "== copying _canneal_report"
cp -pR "$REPORT_DIR" "$DEST/_canneal_report"

echo "== copying SLURM logs"
# Logs do not live under $RUNS, so the verifier cannot infer where each came
# from.  Record the mapping as it is copied and verify against THAT, rather
# than exempting the logs from verification.
mkdir -p "$DEST/_slurm_logs"
: > "$DEST/_slurm_logs/_SOURCES.tsv"
shopt -s nullglob
for L in "$REPO"/l3_canneal_*.out "$REPO"/l3_canneal_*.err \
         "$RUNS"/l3_canneal_*.out "$RUNS"/l3_canneal_*.err \
         ${L3_SLURM_LOG_DIR:+"$L3_SLURM_LOG_DIR"/l3_canneal_*}; do
    JOBID=$(basename "$L" | sed -E 's/^l3_canneal_([0-9]+)_.*/\1/')
    mkdir -p "$DEST/_slurm_logs/$JOBID"
    cp -p "$L" "$DEST/_slurm_logs/$JOBID/"
    printf '%s\t%s\n' "_slurm_logs/$JOBID/$(basename "$L")" "$L" \
        >> "$DEST/_slurm_logs/_SOURCES.tsv"
done
shopt -u nullglob
NLOGS=$(find "$DEST/_slurm_logs" -type f | wc -l | tr -d ' ')
echo "   $NLOGS SLURM log file(s) archived"
if (( NLOGS == 0 )); then
    echo "   NOTE: none found automatically; export L3_SLURM_LOG_DIR and rerun"
    echo "         with a fresh L3_ARCHIVE_NAME if the logs must be included."
fi

# ---- manifest -------------------------------------------------------------
echo "== writing ARCHIVE_MANIFEST.json"
L3_DEST="$DEST" L3_SRC="$RUNS" L3_COMMIT="$COMMIT" L3_BRANCH="$BRANCH" \
L3_BLOB="$BLOB" L3_GLOVE_SHA="$GLOVE_SHA" L3_CPOP="$C_POP_SHA" \
L3_NPOP="$N_POP_SHA" L3_ENDPOINT="$ENDPOINT" L3_SRCCK="$SOURCE_CKPT" \
L3_INCSRC="$INCLUDE_SOURCE" python3 - <<'PY'
import hashlib, json, os, subprocess
D, S = os.environ["L3_DEST"], os.environ["L3_SRC"]
end, srcck = os.environ["L3_ENDPOINT"], os.environ["L3_SRCCK"]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
runs = sorted(x for x in os.listdir(D)
              if x.startswith("final_canneal_"))
endpoints = {r: sha(os.path.join(D, r, "checkpoints", end)) for r in runs}
lineage = {}
lin = os.path.join(D, "_lineage_u1400")
if os.path.isdir(lin):
    for r in sorted(os.listdir(lin)):
        p = os.path.join(lin, r, "checkpoints", srcck)
        if os.path.exists(p):
            lineage[r] = sha(p)
nfiles = sum(len(f) for _, _, f in os.walk(D))
nbytes = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fs in os.walk(D) for f in fs)
json.dump({
  "archive": "canneal_u1400_to_u2000",
  "created_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                capture_output=True, text=True).stdout.strip(),
  "source_root_read_only": os.path.abspath(S),
  "git": {"commit": os.environ["L3_COMMIT"], "branch": os.environ["L3_BRANCH"],
          "train_joint_scratch_blob": os.environ["L3_BLOB"]},
  "hashes": {"glove_sha256": os.environ["L3_GLOVE_SHA"],
             "comprehension_population_sha256": os.environ["L3_CPOP"],
             "naming_population_sha256": os.environ["L3_NPOP"],
             "comprehension_population_n": 27981,
             "naming_population_n": 29571,
             "repetition_population_n": 29571},
  "experiment": {
     "schedule": "interleaved_123", "ratio": [1, 2, 3],
     "u_from": 1400, "u_to": 2000,
     "source_step": 3889200, "end_step": 5556000,
     "milestones": 24, "milestone_every_u": 25,
     "lr_repetition": 3e-5, "lr_naming": 3e-5,
     "arms": {"ctrl": 1e-4, "5e5": 5e-5, "3e5": 3e-5},
     "seeds": [19, 20, 21, 22],
     "end_cursors": {"repetition": 926000, "naming": 1852000,
                     "comprehension": 2778000, "pool": 926000},
     "end_exposures": {"repetition": 2000, "naming": 4000,
                       "comprehension": 6342.4658}},
  "result_u2000": {
     "ctrl_c_errors": [52, 62, 51, 56], "ctrl_c_errors_mean": 55.25,
     "5e5_c_errors": [57, 67, 61, 67], "5e5_c_errors_mean": 63.0,
     "3e5_c_errors": [63, 69, 61, 70], "3e5_c_errors_mean": 65.75,
     "paired_delta_vs_ctrl": {"5e5": 7.75, "3e5": 10.5},
     "ctrl_r_errors_mean": 3.0, "ctrl_n_errors_mean": 0,
     "ctrl_ltm_repetition_mean": 0.8698,
     "cursor_identity_rows": 96, "cursor_identity_failures": 0,
     "strict_ceiling_hit": False,
     "best_arm_for_comprehension": "ctrl (C LR 1e-4)"},
  "endpoint_sha256": endpoints,
  "lineage_u1400_sha256": lineage,
  "include_source_checkpoints": bool(int(os.environ["L3_INCSRC"])),
  "n_files": nfiles, "n_bytes": nbytes},
  open(os.path.join(D, "ARCHIVE_MANIFEST.json"), "w"), indent=1)
print(f"   {len(endpoints)} endpoints, {len(lineage)} lineage checkpoints, "
      f"{nfiles} files, {nbytes/1e6:.1f} MB")
PY

# ---- top-level manifest ---------------------------------------------------
echo "== writing top-level SHA256SUMS"
( cd "$DEST" && find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS )
echo "   $(wc -l < "$DEST/SHA256SUMS" | tr -d ' ') entries"

# ---- verification: destination vs SOURCE, file by file -------------------
echo "== verifying every archived file against its source"
L3_DEST="$DEST" L3_SRC="$RUNS" L3_ENDPOINT="$ENDPOINT" python3 - <<'PY'
import hashlib, os, sys
D, S = os.environ["L3_DEST"], os.environ["L3_SRC"]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# map every archived file back to the path it came from
GENERATED = {"SHA256SUMS", "ARCHIVE_MANIFEST.json", "_SOURCES.tsv"}
LOGSRC = {}
lp = os.path.join(D, "_slurm_logs", "_SOURCES.tsv")
if os.path.exists(lp):
    for line in open(lp, encoding="utf-8"):
        if line.strip():
            rel, src = line.rstrip("\n").split("\t", 1)
            LOGSRC[rel.replace("/", os.sep)] = src

def source_of(rel):
    parts = rel.split(os.sep)
    if parts[0] == "_lineage_u1400":
        return os.path.join(S, *parts[1:])
    if parts[0] == "_canneal_report":
        return os.path.join(S, *parts)
    if parts[0] == "_slurm_logs":
        return LOGSRC.get(rel)           # recorded when the log was copied
    return os.path.join(S, rel)

checked = mismatch = unmapped = notfound = skipped = 0
for dp, _, fs in os.walk(D):
    for f in fs:
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, D)
        if f in GENERATED:
            skipped += 1
            continue
        src = source_of(rel)
        if src is None:
            print(f"  UNMAPPED (cannot be verified): {rel}")
            unmapped += 1
            continue
        if not os.path.exists(src):
            print(f"  SOURCE GONE: {rel}")
            notfound += 1
            continue
        if sha(p) != sha(src):
            print(f"  MISMATCH: {rel}")
            mismatch += 1
        checked += 1

# and the reverse direction for the payload that must be complete
missing_dest = 0
for r in sorted(x for x in os.listdir(D) if x.startswith("final_canneal_")):
    for f in ("metrics.tsv", "config.json", "provenance.json",
              os.path.join("checkpoints", os.environ["L3_ENDPOINT"])):
        if not os.path.exists(os.path.join(D, r, f)):
            print(f"  ABSENT IN ARCHIVE: {r}/{f}")
            missing_dest += 1

print(f"  compared {checked} file(s) byte-for-byte against their source")
print(f"  generated in place, no source by construction: {skipped}")
if mismatch or notfound or missing_dest or unmapped:
    print(f"  FAILURES: {mismatch} mismatched, {notfound} source-gone, "
          f"{missing_dest} absent from archive, {unmapped} unmappable")
    sys.exit(1)
sys.exit(0)
PY
VERIFY_RC=$?

# ---- self-consistency of the manifests we just wrote ---------------------
echo "== re-checking the archive against its own SHA256SUMS"
( cd "$DEST" && sha256sum -c --quiet SHA256SUMS ) || VERIFY_RC=1
for R in "${ALL_RUNS[@]}"; do
    ( cd "$DEST/$R" && sha256sum -c --quiet SHA256SUMS ) || VERIFY_RC=1
done
if (( INCLUDE_SOURCE )); then
    ( cd "$DEST/_lineage_u1400" && sha256sum -c --quiet SHA256SUMS ) \
        || VERIFY_RC=1
fi

if (( VERIFY_RC != 0 )); then
    echo "CANNEAL_STORE_VERIFIED=NO"
    echo "FATAL: verification failed; the archive is NOT trustworthy."
    echo "       Nothing was removed from SCRATCH.  Inspect $DEST and rerun"
    echo "       with a fresh L3_ARCHIVE_NAME."
    exit 1
fi

# ---- immutability ---------------------------------------------------------
chmod -R a-w "$DEST"
echo "== archive made read-only (a-w)"

echo ""
echo "=========================================================="
echo "STORE PATH: $DEST"
echo "commit:     $COMMIT"
echo "blob:       $BLOB"
echo "files:      $(find "$DEST" -type f | wc -l | tr -d ' ')"
echo "size:       $(du -sh "$DEST" | cut -f1)"
echo "SCRATCH was read only; nothing was deleted."
echo "CANNEAL_STORE_VERIFIED=YES"
echo "=========================================================="
