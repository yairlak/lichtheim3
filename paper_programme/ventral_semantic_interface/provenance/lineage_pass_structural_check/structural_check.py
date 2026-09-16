"""READ-ONLY provenance structural check. NO decoding, NO scoring, NO S0-S3."""
import csv, hashlib, json, os, sys

W = "/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-gate-x-lesion"
R = "/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3"
os.chdir(W)
sys.path.insert(0, W)
import torch

from gate_x_lesion.identity import reconstructed_state_sha256
from scripts.gating_diagnostics.run_gate_route_audit import build_state, sha256_file, load_manifest
from scripts.naming_comprehension.frozen_head_probe import sha256_tensor

out = {"torch": torch.__version__, "python": sys.version.split()[0]}

# ---- data files
glove_link = os.path.join(W, "data", "glove.6B.300d.txt")
out["glove"] = {"link": glove_link, "realpath": os.path.realpath(glove_link),
                "bytes": os.path.getsize(glove_link), "sha256": sha256_file(glove_link)}
for name in ("lexicon_en_glove_covered.tsv", "lexicon_en.tsv"):
    p = os.path.join(W, "data", name)
    out[name] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}

man = {r["state_id"]: r for r in load_manifest(
    os.path.join(W, "paper_programme/gating_route_diagnostics/checkpoint_manifest.tsv"))}
prop = {r["state_id"]: r for r in csv.DictReader(open(os.path.join(
    W, "paper_programme/gate_x_lesion_recovery/checkpoint_manifest.proposed.tsv")), delimiter="\t")}

EXPECT_KEYS = {"2.weight", "2.bias"}
for w in ("W3", "W4"):
    src, rep = man[f"{w}_SRC"], man[f"{w}_REP"]
    rec = {}
    ck_path = os.path.join(R, src["artifact_path"])
    hd_path = os.path.join(R, rep["applies_head_path"])
    before = {"ckpt": sha256_file(ck_path), "head": sha256_file(hd_path)}
    rec["sha_before"] = before
    rec["manifest_pair_same_base"] = src["artifact_sha256"] == rep["artifact_sha256"]

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    rec["ckpt_top_keys"] = sorted(ck.keys())
    rec["ckpt_meta"] = {k: ck.get(k) for k in (
        "seed", "global_step", "regime", "schedule", "subset_mode", "lexicon_path",
        "lexicon_file_sha256", "glove_present", "n_glove_found", "n_glove_fallback",
        "widths", "comprehension_population_sha256", "optimizer_policy", "dec_weight",
        "c_align_weight") if k in ck}
    rec["ckpt_config_ltm"] = ck["config"]["ltm"]
    rec["ckpt_config_wm"] = ck["config"]["wm"]
    rec["ckpt_config_gating"] = ck["config"]["gating"]
    rec["ckpt_config_data_semantic_dim"] = ck["config"]["data"].get("semantic_dim")
    rec["ckpt_config_data_glove_path"] = ck["config"]["data"].get("glove_path")
    sd_key = [k for k in ck if "model" in k.lower()]
    rec["model_state_keys_in_ckpt"] = sd_key

    head = torch.load(hd_path, map_location="cpu", weights_only=False)
    rec["head_top_keys"] = sorted(head.keys())
    rec["head_meta"] = {k: (v if not isinstance(v, (torch.Tensor, dict)) else "<tensor/dict>")
                        for k, v in head.items() if k != "state"}
    rec["head_state"] = {k: [list(v.shape), str(v.dtype)] for k, v in head["state"].items()}
    rec["head_keys_exact"] = set(head["state"]) == EXPECT_KEYS

    # ---- canonical reconstruction (same function the GATING/GXLR runners use)
    tr, m_src, prov_s, _, _ = build_state(src, "cpu")
    _, m_rep, prov_r, _, _ = build_state(rep, "cpu")
    sd_s = m_src.state_dict(); sd_r = m_rep.state_dict()
    live_sd = tr.model.state_dict()
    rec["n_params_tensors"] = len(sd_s)
    rec["same_keyset"] = set(sd_s) == set(sd_r)
    rec["shapes_match"] = all(sd_s[k].shape == sd_r[k].shape for k in sd_s)
    rec["differing_tensors"] = sorted(k for k in sd_s if not torch.equal(sd_s[k], sd_r[k]))
    rec["head_equals_rep_param"] = all(torch.equal(head["state"][k], sd_r["ltm.to_semantic." + k])
                                       for k in head["state"])
    groups = {"ventral_decoder(sem_to_h0,decoder,dec_to_premotor)": ("ltm.sem_to_h0.", "ltm.decoder.", "ltm.dec_to_premotor."),
              "ltm_encoder": ("ltm.encoder.",), "to_semantic.0": ("ltm.to_semantic.0.",),
              "dorsal_wm": ("wm.",), "gate": ("gate.",), "motor": ("motor.",), "phon_embed": ("phon_embed.",)}
    rec["group_identical"] = {g: all(torch.equal(sd_s[k], sd_r[k]) for k in sd_s if k.startswith(p))
                              for g, p in groups.items()}
    rec["gate_param_tensors"] = sorted(k for k in sd_s if k.startswith("gate."))
    rec["semantic_bank_buffer_in_state_dict"] = any("semantic_bank" in k for k in sd_s)
    bank_s, bank_r = m_src.ltm.semantic_bank, m_rep.ltm.semantic_bank
    rec["semantic_bank_shape"] = list(bank_s.shape)
    rec["semantic_bank_identical_src_rep"] = torch.equal(bank_s, bank_r)
    rec["semantic_bank_row_norm_min_max"] = [float(bank_s.norm(dim=1).min()), float(bank_s.norm(dim=1).max())]
    rec["bank_raw_shape"] = list(tr.bank_raw.shape)
    rec["bank_raw_sha256_tensor"] = sha256_tensor(tr.bank_raw)
    rec["bank_raw_norm_min_mean_max"] = [float(tr.bank_raw.norm(dim=1).min()),
                                         float(tr.bank_raw.norm(dim=1).mean()),
                                         float(tr.bank_raw.norm(dim=1).max())]
    rec["bank_equals_normalize_bank_raw"] = torch.equal(
        bank_s, torch.nn.functional.normalize(tr.bank_raw, dim=-1))
    rec["src_to_semantic_2_weight_sha256_tensor"] = sha256_tensor(sd_s["ltm.to_semantic.2.weight"])
    rec["src_to_semantic_2_bias_sha256_tensor"] = sha256_tensor(sd_s["ltm.to_semantic.2.bias"])
    rec["src_sem_to_h0_weight_sha256_tensor"] = sha256_tensor(sd_s["ltm.sem_to_h0.weight"])
    rec["src_to_semantic_0_weight_sha256_tensor"] = sha256_tensor(sd_s["ltm.to_semantic.0.weight"])
    rec["rep_to_semantic_2_weight_sha256_tensor"] = sha256_tensor(sd_r["ltm.to_semantic.2.weight"])
    rec["live_trainer_model_untouched_by_isolation"] = all(torch.equal(live_sd[k], sd_s[k]) for k in sd_s)
    rec["n_entries"] = len(tr.entries)
    rec["n_comp_idx"] = len(tr.comp_idx)
    rec["comp_hash"] = tr.comp_hash
    rec["glove_found_fallback"] = [tr.glove_found, tr.glove_fallback]
    rec["max_form_len"] = max(len(e.phonemes) for e in tr.entries)
    rec["vocab"] = {"size": tr.vocab.size, "pad": tr.vocab.pad_id, "bos": tr.vocab.bos_id, "eos": tr.vocab.eos_id,
                    "itos_sha256": hashlib.sha256("\n".join(tr.vocab.itos).encode()).hexdigest()}
    rec["item_order_sha256_all_entries"] = hashlib.sha256(",".join(e.word for e in tr.entries).encode()).hexdigest()
    rec["lexicon_load_stats"] = {k: str(v) for k, v in vars(tr.lexicon.load_stats).items()}
    rec["prov_src"] = {k: prov_s[k] for k in ("global_step", "seed", "lexicon_file_sha256", "gating_config")}
    rec["prov_rep_head"] = {k: prov_r.get(k) for k in ("head_sha256", "head_label", "head_arm")}
    rec["model_training_flag"] = [m_src.training, m_rep.training]

    comp_src = reconstructed_state_sha256(src["artifact_sha256"], "")
    comp_rep = reconstructed_state_sha256(rep["artifact_sha256"], rep["applies_head_sha256"])
    comp_rep_live = reconstructed_state_sha256(before["ckpt"], before["head"])
    rec["gxlr_state_v1"] = {"SRC": comp_src, "REP": comp_rep, "REP_from_live_hashes": comp_rep_live,
                            "proposed_manifest_SRC": prop[f"{w}_SRC"]["state_sha256"],
                            "proposed_manifest_REP": prop[f"{w}_REP"]["state_sha256"]}
    rec["sha_after"] = {"ckpt": sha256_file(ck_path), "head": sha256_file(hd_path)}
    rec["unchanged"] = rec["sha_after"] == before
    out[w] = rec
    del tr, m_src, m_rep, ck, head

print(json.dumps(out, indent=1, default=str))
