"""CAP-3: route-isolated capacity probes for COMPREHENSION and DORSAL width.

CAP-1/CAP-2 established the ventral PRODUCTION result: full-lexicon naming is
solved at LTM decoder width 512 (exact 1.0 over all 29,571 words at 125
exposures, lr 1e-3 -> 1e-4 at 100).  Yair asked for "the same search for
Ventral and dorsal", so this driver runs the identical width protocol
{128, 256, 512} on the two routes the naming probe did NOT test:

  --route comprehension : varies ltm.enc_hidden (phonology -> encoder ->
      to_semantic -> s_hat -> cosine retrieval against the full 29,571 bank).
      The production decoder never runs.
  --route dorsal        : varies wm.hidden (phonology -> WM encoder -> WM
      decoder -> to_premotor -> motor).  The LTM route never runs, so lexical
      retrieval cannot solve repetition for the WM route.

THE THREE WIDTHS ARE INDEPENDENT.  --wm-hidden / --enc-hidden / --dec-hidden
are set separately and recorded separately; a probe changes exactly one and
the other two stay at the canonical 128 (naming's 512 decoder is NOT carried
in, because that would confound the comparison).  Tests assert this.

Shared across every width of every route: seed 22, the counter-addressed item
order, AdamW, wd 1e-5, grad clip 1.0, batch 64, teacher forcing, and the
two-stage capacity schedule lr 1e-3 through exposure 100 then 1e-4 -- the
schedule validated by CAP-2, applied in-run here rather than as a resume
branch.  It is a CONTROLLED probe schedule, not a claim of per-route
optimality, and it is never tuned per width.

Losses are the isolated, UNWEIGHTED objective of each route:
  comprehension : retrieval CE only (full-bank cosine/tau, tau = 0.10)
  dorsal        : WM-route sequence CE only
See the module notes in the pre-submission report for why lambda_C = 0.087 is
deliberately NOT carried into the isolated probe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.dual_route import DualRouteModel                              # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import (            # noqa: E402
    EXPECTED_CANONICAL_C_HASH, EXPECTED_CANONICAL_C_N, CounterStream,
    build_batch, canonical_config, capture_rng_states, derive_stream_seeds,
    restore_rng_states,
)
from scripts.naming_comprehension.train_tasks import (                    # noqa: E402
    canonical_phonology_indices, comprehension_forward, retrieval_loss,
    subset_definition_hash, subset_records,
)
from data.lexicon import build_lexicon, logfreq_weights                   # noqa: E402
from data.phonemes import build_vocab                                     # noqa: E402
from utils.provenance import git_state, sha256_file                       # noqa: E402

FORMAT = "lichtheim3.route_capacity.v1"
ROUTE_COMPREHENSION = "comprehension"
ROUTE_DORSAL = "dorsal"
ROUTES = (ROUTE_COMPREHENSION, ROUTE_DORSAL)

CANONICAL_WIDTH = 128              # every width NOT under test stays here
WIDTHS = (128, 256, 512)
LR_STAGE1, LR_STAGE2 = 1e-3, 1e-4
LR_BOUNDARY_EXPOSURES = 100        # CAP-2's validated transition point
WEIGHT_DECAY, GRAD_CLIP, BATCH_SIZE = 1e-5, 1.0, 64
TAU = 0.10                         # historical validated retrieval temperature

# Genuine free-AR cap for the SECONDARY dorsal metric.  A single global
# constant, never an item's own gold length: the longest form in either
# evaluated population is 9 phonemes, so 12 leaves room for EOS plus
# over-generation without truncating a correct answer.
WM_FREE_AR_MAX_STEPS = 12

# Trainable scope per route.  Everything else keeps its init value, asserted
# at every evaluation.
SCOPE = {
    ROUTE_COMPREHENSION: ("phon_embed.", "ltm.phon_embed.", "ltm.encoder.",
                          "ltm.to_semantic."),
    ROUTE_DORSAL: ("phon_embed.", "wm.", "motor."),
}

METRIC_COLUMNS = {
    ROUTE_COMPREHENSION: [
        "step", "exposures", "lr", "train_ce_running", "retrieval_ce",
        "top1", "top5", "rank_mean", "rank_median", "target_cos_mean",
        "margin_mean", "elapsed_s"],
    ROUTE_DORSAL: [
        "step", "exposures", "lr", "train_ce_running", "wm_ce",
        "tf_token_acc", "lex_exact", "lex_exact_freear", "lex_n",
        "pseudo_exact", "pseudo_exact_freear", "pseudo_n",
        "pseudo_exact_short", "pseudo_exact_long",
        "pseudo_exact_freear_short", "pseudo_exact_freear_long",
        "elapsed_s"],
}


def lr_for_exposure(exposures: float) -> float:
    """Two-stage capacity schedule, identical for every route and width."""
    return LR_STAGE1 if exposures < LR_BOUNDARY_EXPOSURES else LR_STAGE2


def param_census(model: DualRouteModel) -> Dict[str, int]:
    g = {"total": 0, "wm": 0, "ltm_encoder": 0, "ltm_to_semantic": 0,
         "ltm_sem_to_h0": 0, "ltm_decoder": 0, "ltm_dec_to_premotor": 0,
         "phon_embed": 0, "motor": 0, "gate": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        g["total"] += n
        if name.startswith("wm."):
            g["wm"] += n
        elif name.startswith("ltm.encoder"):
            g["ltm_encoder"] += n
        elif name.startswith("ltm.to_semantic"):
            g["ltm_to_semantic"] += n
        elif name.startswith("ltm.sem_to_h0."):
            g["ltm_sem_to_h0"] += n
        elif name.startswith("ltm.decoder."):
            g["ltm_decoder"] += n
        elif name.startswith("ltm.dec_to_premotor."):
            g["ltm_dec_to_premotor"] += n
        elif name.startswith(("phon_embed.", "ltm.phon_embed.")):
            g["phon_embed"] += n
        elif name.startswith("motor."):
            g["motor"] += n
        elif name.startswith("gate."):
            g["gate"] += n
    return g


def load_pseudowords(vocab, path: str = "data/eval_external/wfe_eval.tsv"
                     ) -> List[dict]:
    """The committed WFE pseudoword population, phonemised with the vocab."""
    import csv
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return []
    rows = []
    with open(full, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("lexicality") != "pseudo":
                continue
            toks = r["target_phonemes"].split()
            if any(t not in vocab.stoi for t in toks):
                continue
            rows.append({"word": r["word"],
                         "phonemes": [vocab.stoi[t] for t in toks],
                         "length": len(toks)})
    return rows


class RouteCapacityTrainer:
    def __init__(self, *, route: str, wm_hidden: int, enc_hidden: int,
                 dec_hidden: int, seed: int, device: str,
                 max_words: int = 30000, batch_size: int = BATCH_SIZE,
                 lexicon_path: str = "data/lexicon_en_glove_covered.tsv",
                 glove_path: Optional[str] = "data/glove.6B.300d.txt",
                 allow_glove_fallback: bool = False,
                 require_population_hash: bool = True) -> None:
        if route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
        self.route = route
        self.seed = int(seed)
        self.device = device
        torch.manual_seed(self.seed)

        cfg = canonical_config(self.seed, device, max_words=max_words,
                               lexicon_path=lexicon_path,
                               dorsal_pool_size=4000, batch_size=batch_size,
                               glove_path=glove_path)
        # THREE INDEPENDENT WIDTHS.  canonical_config pins all three to 128;
        # each is overridden explicitly so no route can be resized silently.
        cfg.wm.hidden = int(wm_hidden)
        cfg.ltm.enc_hidden = int(enc_hidden)
        cfg.ltm.dec_hidden = int(dec_hidden)
        cfg.ltm.__post_init__()
        self.cfg = cfg
        self.widths = {"wm_hidden": int(wm_hidden),
                       "ltm_enc_hidden": int(enc_hidden),
                       "ltm_dec_hidden": int(dec_hidden)}

        self.vocab = build_vocab()
        self.lexicon = build_lexicon(cfg.data, self.vocab)
        self.entries = list(self.lexicon.entries)
        stats = self.lexicon.load_stats
        self.glove_fallback = int(getattr(stats, "n_glove_fallback", 0))
        if self.glove_fallback and not allow_glove_fallback:
            raise RuntimeError(
                f"{self.glove_fallback} entries lack real GloVe; the capacity "
                f"probe requires real vectors")
        self.bank_raw = torch.stack(
            [torch.tensor(e.semantic) for e in self.entries]).float()

        self.model = DualRouteModel(cfg, self.vocab).to(device)
        self.model.set_semantic_bank(self.bank_raw.to(device))

        # ---- populations -------------------------------------------------
        if route == ROUTE_COMPREHENSION:
            # canonical one-target-per-phonology; bank stays the FULL lexicon
            self.train_idx = canonical_phonology_indices(self.entries)
            self.population_hash = subset_definition_hash(
                subset_records(self.entries, self.train_idx, self.vocab))
            if require_population_hash:
                if len(self.train_idx) != EXPECTED_CANONICAL_C_N:
                    raise RuntimeError(
                        f"canonical C population is {len(self.train_idx)}, "
                        f"expected {EXPECTED_CANONICAL_C_N}")
                if self.population_hash != EXPECTED_CANONICAL_C_HASH:
                    raise RuntimeError(
                        f"canonical C hash {self.population_hash} != expected "
                        f"{EXPECTED_CANONICAL_C_HASH}")
            stream_name = "comprehension"
        else:
            self.train_idx = list(range(len(self.entries)))
            self.population_hash = subset_definition_hash(
                subset_records(self.entries, self.train_idx, self.vocab))
            stream_name = "repetition"
        self.stream_name = stream_name

        seeds = derive_stream_seeds(self.seed)
        # SAMPLER IS ROUTE-SPECIFIC, matching each route's own canonical
        # recipe.  Cross-task equality of the sampler is not required: the
        # scientific comparison is WITHIN a route, across widths, under one
        # identical protocol.
        #
        #   dorsal        : the canonical/historical repetition sampler that
        #       established the existing WM128 ceiling -- log-frequency
        #       weights (log((N+1)/rank)) raised to freq_temp = 1.0, clipped
        #       at 1e-6, normalised, drawn WITH REPLACEMENT via multinomial,
        #       463 batches per pass over all 29,571 items.
        #   comprehension : unweighted permutation, the canonical C stream.
        if route == ROUTE_DORSAL:
            w = logfreq_weights([self.entries[i].rank
                                 for i in self.train_idx]) ** float(
                                     self.cfg.data.freq_temp)
            w = np.clip(w, 1e-6, None)
            weights = w / w.sum()
            self.sampler_note = (
                f"log-frequency weighted, with replacement, "
                f"freq_temp={self.cfg.data.freq_temp} (canonical historical "
                f"repetition sampler)")
        else:
            weights = None
            self.sampler_note = "unweighted permutation per pass (canonical C)"
        self.stream = CounterStream(stream_name, self.train_idx, batch_size,
                                    seeds[stream_name], weights=weights)
        self.per_epoch = self.stream.per_epoch

        self.pseudo = (load_pseudowords(self.vocab)
                       if route == ROUTE_DORSAL else [])

        # ---- trainable scope --------------------------------------------
        self.scope = SCOPE[route]
        trainable, frozen = [], []
        for name, p in self.model.named_parameters():
            if name.startswith(self.scope):
                trainable.append(name)
            else:
                p.requires_grad_(False)
                frozen.append(name)
        self.trainable_names = sorted(set(trainable))
        self.frozen_names = sorted(set(frozen))
        self.frozen_ref = {n: p.detach().clone().cpu()
                           for n, p in self.model.named_parameters()
                           if not n.startswith(self.scope)}

        self.optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
        self.global_step = 0
        self.running_ce: Optional[float] = None

    # ------------------------------------------------------------------ util
    @property
    def exposures(self) -> float:
        return self.global_step / self.per_epoch

    def current_lr(self) -> float:
        return lr_for_exposure(self.exposures)

    def assert_frozen(self) -> None:
        params = dict(self.model.named_parameters())
        moved = [n for n, ref in self.frozen_ref.items()
                 if not torch.equal(params[n].detach().cpu(), ref)]
        if moved:
            raise RuntimeError(
                f"{self.route} probe moved out-of-scope parameters: {moved[:8]}")

    # ----------------------------------------------------------------- train
    def loss_on(self, batch: dict) -> torch.Tensor:
        if self.route == ROUTE_COMPREHENSION:
            s_hat = comprehension_forward(self.model, batch["enc_in"],
                                          batch["enc_mask"])
            # UNWEIGHTED retrieval CE (lambda = 1); no alignment term.
            return retrieval_loss(s_hat, self.model.ltm.semantic_bank,
                                  batch["bank_idx"], TAU)
        # dorsal: WM-route sequence CE only, LTM never called
        h = self.model.wm.encode(batch["enc_in"], batch["enc_mask"])
        premotor = self.model.wm.decode_from_state(h, batch["dec_in"])["premotor"]
        logits = self.model.motor(premotor)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["dec_tgt"].reshape(-1),
            ignore_index=self.vocab.pad_id)

    def train_step(self) -> float:
        self.model.train()
        lr = self.current_lr()
        for g in self.optim.param_groups:
            g["lr"] = lr
        batch = build_batch(self.entries, self.bank_raw, self.vocab,
                            self.stream.indices(self.global_step), self.device)
        loss = self.loss_on(batch)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], GRAD_CLIP)
        self.optim.step()
        self.global_step += 1
        ce = float(loss.detach())
        self.running_ce = (ce if self.running_ce is None
                           else 0.99 * self.running_ce + 0.01 * ce)
        return ce

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self) -> Dict[str, object]:
        self.model.eval()
        if self.route == ROUTE_COMPREHENSION:
            return self._evaluate_comprehension()
        return self._evaluate_dorsal()

    def _evaluate_comprehension(self) -> Dict[str, object]:
        from scripts.naming_comprehension.frozen_probe import (
            comprehension_metrics, encode_all)
        forms = [self.entries[i].phonemes for i in self.train_idx]
        s_hat = encode_all(self.model, self.vocab, forms, self.device, 512)
        m = comprehension_metrics(s_hat.cpu(), self.bank_raw.cpu(),
                                  self.train_idx, 512)
        ce_sum = ce_n = 0
        for lo in range(0, len(self.train_idx), 512):
            idx = self.train_idx[lo:lo + 512]
            b = build_batch(self.entries, self.bank_raw, self.vocab, idx,
                            self.device)
            ce_sum += float(self.loss_on(b)) * len(idx)
            ce_n += len(idx)
        return {"retrieval_ce": ce_sum / max(ce_n, 1),
                "top1": float(np.mean(m["top1"])),
                "top5": float(np.mean(m["top5"])),
                "rank_mean": float(np.mean(m["target_rank"])),
                "rank_median": float(np.median(m["target_rank"])),
                "target_cos_mean": float(np.mean(m["target_cos"])),
                "margin_mean": float(np.mean(m["margin"])),
                "_ceiling": float(np.mean(m["top1"])) == 1.0}

    def _wm_exact(self, forms: Sequence[List[int]], batch_size: int = 256
                  ) -> List[int]:
        """Route-isolated WM autoregressive repetition, canonical convention.

        Canonical repetition AR is FORCED-LENGTH (the repo's
        evaluate_forms_ar): the greedy loop runs to the batch maximum and each
        item is truncated to its own gold length + 1, then cut at the first
        EOS.  Legitimate for repetition -- the form IS the input -- and it is
        the convention every historical repetition number uses.
        """
        ok: List[int] = []
        for lo in range(0, len(forms), batch_size):
            chunk = list(forms[lo:lo + batch_size])
            max_enc = max(len(f) for f in chunk) + 1
            enc_in = torch.full((len(chunk), max_enc), self.vocab.pad_id,
                                dtype=torch.long)
            enc_mask = torch.zeros((len(chunk), max_enc), dtype=torch.bool)
            for i, f in enumerate(chunk):
                enc_in[i, :len(f) + 1] = torch.tensor(f + [self.vocab.eos_id])
                enc_mask[i, :len(f) + 1] = True
            enc_in, enc_mask = enc_in.to(self.device), enc_mask.to(self.device)
            h = self.model.wm.encode(enc_in, enc_mask)      # eval: no noise
            dec = torch.full((len(chunk), 1), self.vocab.bos_id,
                             dtype=torch.long, device=self.device)
            for _ in range(max(len(f) for f in chunk) + 1):
                pre = self.model.wm.decode_from_state(h, dec)["premotor"]
                nxt = self.model.motor(pre)[:, -1, :].argmax(-1, keepdim=True)
                dec = torch.cat([dec, nxt], dim=1)
            for i, f in enumerate(chunk):
                seq = dec[i, 1:len(f) + 2].tolist()
                if self.vocab.eos_id in seq:
                    seq = seq[:seq.index(self.vocab.eos_id)]
                ok.append(int(seq == f))
        return ok

    def _wm_exact_free(self, forms: Sequence[List[int]],
                       batch_size: int = 256) -> List[int]:
        """SECONDARY readout: genuinely free WM autoregressive repetition.

        Identical to `_wm_exact` except that the greedy loop runs to a single
        GLOBAL cap (WM_FREE_AR_MAX_STEPS) and the prediction is cut at the
        first EOS wherever it falls.  The item's own gold length is never
        consulted, so an over-long or never-terminating output counts as an
        error instead of being silently truncated to the right length.

        Reported ALONGSIDE the canonical forced-length metric, never instead
        of it: historical repetition numbers keep their own convention and
        are not restated here.
        """
        ok: List[int] = []
        for lo in range(0, len(forms), batch_size):
            chunk = list(forms[lo:lo + batch_size])
            max_enc = max(len(f) for f in chunk) + 1
            enc_in = torch.full((len(chunk), max_enc), self.vocab.pad_id,
                                dtype=torch.long)
            enc_mask = torch.zeros((len(chunk), max_enc), dtype=torch.bool)
            for i, f in enumerate(chunk):
                enc_in[i, :len(f) + 1] = torch.tensor(f + [self.vocab.eos_id])
                enc_mask[i, :len(f) + 1] = True
            enc_in, enc_mask = enc_in.to(self.device), enc_mask.to(self.device)
            h = self.model.wm.encode(enc_in, enc_mask)      # eval: no noise
            dec = torch.full((len(chunk), 1), self.vocab.bos_id,
                             dtype=torch.long, device=self.device)
            for _ in range(WM_FREE_AR_MAX_STEPS):
                pre = self.model.wm.decode_from_state(h, dec)["premotor"]
                nxt = self.model.motor(pre)[:, -1, :].argmax(-1, keepdim=True)
                dec = torch.cat([dec, nxt], dim=1)
                if bool((dec == self.vocab.eos_id).any(dim=1).all()):
                    break
            for i, f in enumerate(chunk):
                seq = dec[i, 1:].tolist()
                if self.vocab.eos_id in seq:
                    seq = seq[:seq.index(self.vocab.eos_id)]
                ok.append(int(seq == f))
        return ok

    def _split_by_length(self, flags: Sequence[int], lens: np.ndarray,
                         med: float) -> Dict[str, float]:
        a = np.asarray(flags)
        return {"short": float(np.mean(a[lens <= med])),
                "long": float(np.mean(a[lens > med]))}

    def _evaluate_dorsal(self) -> Dict[str, object]:
        forms = [e.phonemes for e in self.entries]
        lex = self._wm_exact(forms)
        lex_free = self._wm_exact_free(forms)
        out = {"lex_exact": float(np.mean(lex)),
               "lex_exact_freear": float(np.mean(lex_free)),
               "lex_n": len(lex)}
        ce_sum = ce_n = tf_ok = tf_all = 0
        for lo in range(0, len(self.train_idx), 512):
            idx = self.train_idx[lo:lo + 512]
            b = build_batch(self.entries, self.bank_raw, self.vocab, idx,
                            self.device)
            ce_sum += float(self.loss_on(b)) * len(idx)
            ce_n += len(idx)
            h = self.model.wm.encode(b["enc_in"], b["enc_mask"])
            pre = self.model.wm.decode_from_state(h, b["dec_in"])["premotor"]
            pred = self.model.motor(pre).argmax(-1)
            m = b["dec_tgt"] != self.vocab.pad_id
            tf_ok += int((pred[m] == b["dec_tgt"][m]).sum())
            tf_all += int(m.sum())
        out["wm_ce"] = ce_sum / max(ce_n, 1)
        out["tf_token_acc"] = tf_ok / max(tf_all, 1)
        if self.pseudo:
            pf = [p["phonemes"] for p in self.pseudo]
            pk, pk_free = self._wm_exact(pf), self._wm_exact_free(pf)
            lens = np.array([p["length"] for p in self.pseudo])
            med = float(np.median(lens))
            can = self._split_by_length(pk, lens, med)
            fre = self._split_by_length(pk_free, lens, med)
            out.update({
                "pseudo_exact": float(np.mean(pk)),
                "pseudo_exact_freear": float(np.mean(pk_free)),
                "pseudo_n": len(pk),
                "pseudo_exact_short": can["short"],
                "pseudo_exact_long": can["long"],
                "pseudo_exact_freear_short": fre["short"],
                "pseudo_exact_freear_long": fre["long"]})
        else:
            for k in ("pseudo_exact", "pseudo_exact_freear",
                      "pseudo_exact_short", "pseudo_exact_long",
                      "pseudo_exact_freear_short", "pseudo_exact_freear_long"):
                out[k] = ""
            out["pseudo_n"] = 0
        # Ceiling stays the CANONICAL metric, for comparability with every
        # historical repetition number.
        out["_ceiling"] = out["lex_exact"] == 1.0
        return out

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        return {"format": FORMAT, "route": self.route, "widths": self.widths,
                "seed": self.seed, "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "running_ce": self.running_ce, "per_epoch": self.per_epoch,
                "lr": self.current_lr(), "tau": TAU,
                "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                "batch_size": self.stream.batch_size,
                "population_n": len(self.train_idx),
                "population_sha256": self.population_hash,
                "trainable_scope": list(self.scope),
                "rng_states": capture_rng_states(), "git": git_state(ROOT)}

    def load_state_dict(self, ckpt: dict) -> None:
        if ckpt.get("format") != FORMAT:
            raise RuntimeError(f"unexpected format {ckpt.get('format')!r}")
        if ckpt["route"] != self.route or ckpt["widths"] != self.widths:
            raise RuntimeError(
                f"checkpoint is {ckpt['route']} {ckpt['widths']}; this trainer "
                f"is {self.route} {self.widths}")
        if ckpt["population_sha256"] != self.population_hash:
            raise RuntimeError("population changed between runs")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = int(ckpt["global_step"])
        self.running_ce = ckpt.get("running_ce")
        restore_rng_states(ckpt["rng_states"])
        self.frozen_ref = {n: p.detach().clone().cpu()
                           for n, p in self.model.named_parameters()
                           if not n.startswith(self.scope)}


# ===========================================================================

def append_row(path: str, row: dict, columns: List[str]) -> None:
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("\t".join(columns) + "\n")
        f.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--route", required=True, choices=list(ROUTES))
    ap.add_argument("--wm-hidden", type=int, default=CANONICAL_WIDTH)
    ap.add_argument("--enc-hidden", type=int, default=CANONICAL_WIDTH)
    ap.add_argument("--dec-hidden", type=int, default=CANONICAL_WIDTH)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-words", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--eval-exposures", required=True)
    ap.add_argument("--max-exposures", type=int, required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--benchmark", type=int, default=0)
    ap.add_argument("--allow-glove-fallback", action="store_true")
    ap.add_argument("--no-population-hash-check", action="store_true")
    ap.add_argument("--lexicon-path", default="data/lexicon_en_glove_covered.tsv")
    ap.add_argument("--glove-path", default="data/glove.6B.300d.txt")
    ap.add_argument("--log-every", type=int, default=4630)
    ap.add_argument("--no-stop-at-ceiling", action="store_true")
    args = ap.parse_args(argv)

    tr = RouteCapacityTrainer(
        route=args.route, wm_hidden=args.wm_hidden, enc_hidden=args.enc_hidden,
        dec_hidden=args.dec_hidden, seed=args.seed, device=args.device,
        max_words=args.max_words, batch_size=args.batch_size,
        lexicon_path=args.lexicon_path, glove_path=args.glove_path,
        allow_glove_fallback=args.allow_glove_fallback,
        require_population_hash=not args.no_population_hash_check)

    census = param_census(tr.model)
    print(f"[probe] route={args.route} widths={tr.widths} "
          f"pop={len(tr.train_idx)} per_epoch={tr.per_epoch} "
          f"pseudo={len(tr.pseudo)}")
    print(f"[probe] params: {json.dumps(census)}")
    print(f"[probe] trainable: {len(tr.trainable_names)} tensors "
          f"{tr.trainable_names}")

    if args.benchmark:
        for _ in range(5):
            tr.train_step()
        if args.device == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        for _ in range(args.benchmark):
            tr.train_step()
        if args.device == "mps":
            torch.mps.synchronize()
        dt = (time.time() - t0) / args.benchmark
        print(f"[probe] benchmark {args.route} {tr.widths}: {dt:.6f} s/step "
              f"({1/dt:.1f}/s); 1 exposure = {tr.per_epoch} steps = "
              f"{dt*tr.per_epoch:.1f}s; {args.max_exposures} exposures ~= "
              f"{dt*tr.per_epoch*args.max_exposures/3600:.2f} h")
        return 0

    run_dir = os.path.join(args.out_dir, args.run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics = os.path.join(run_dir, "metrics.tsv")
    columns = METRIC_COLUMNS[args.route]

    if args.resume:
        tr.load_state_dict(torch.load(args.resume, map_location=args.device,
                                      weights_only=False))
        print(f"[probe] resumed step {tr.global_step} ({tr.exposures:.1f} exp)")

    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        json.dump({
            "format": FORMAT, "route": args.route, "widths": tr.widths,
            "seed": args.seed, "device": args.device,
            "lr_schedule": f"{LR_STAGE1} through exposure "
                           f"{LR_BOUNDARY_EXPOSURES}, then {LR_STAGE2}",
            "lr_stage1": LR_STAGE1, "lr_stage2": LR_STAGE2,
            "lr_boundary_exposures": LR_BOUNDARY_EXPOSURES,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "batch_size": args.batch_size, "tau": TAU,
            "loss": ("unweighted full-bank retrieval CE (lambda=1, no "
                     "alignment term)" if args.route == ROUTE_COMPREHENSION
                     else "WM-route sequence CE only (LTM never called)"),
            "lambda_note": ("lambda_C=0.087 is the JOINT-task weight on the "
                            "sole C term in train_joint_scratch; it is "
                            "deliberately NOT carried into this isolated "
                            "probe, so probe CE is not comparable to joint "
                            "retrieval_ce values"),
            "population_n": len(tr.train_idx),
            "population_sha256": tr.population_hash,
            "retrieval_bank_n": len(tr.entries),
            "sampler": tr.sampler_note,
            "free_ar_max_steps": WM_FREE_AR_MAX_STEPS,
            "wm_interference_noise": tr.cfg.wm.interference_noise,
            "ltm_encoder_mode": tr.cfg.ltm.ltm_encoder_mode,
            "trainable_scope": list(tr.scope),
            "trainable_parameters": tr.trainable_names,
            "frozen_parameters": tr.frozen_names,
            "params": census,
            "pseudoword_n": len(tr.pseudo),
            "evaluation": ("strict canonical top-1 word ID over the C "
                           "population against the full bank"
                           if args.route == ROUTE_COMPREHENSION else
                           "PRIMARY route-isolated WM autoregressive "
                           "repetition, canonical forced-length convention "
                           "(comparable to all historical numbers); SECONDARY "
                           "genuine free-AR to a global cap that never "
                           "consults gold length"),
            "lexicon_file_sha256": sha256_file(
                os.path.join(ROOT, args.lexicon_path)),
            "max_exposures": args.max_exposures,
            "eval_exposures": args.eval_exposures,
            "git": git_state(ROOT),
        }, open(cfg_path, "w"), indent=1)

    eval_steps = sorted({int(round(float(e) * tr.per_epoch))
                         for e in args.eval_exposures.split(",")})
    max_steps = args.max_exposures * tr.per_epoch
    t0 = time.time()

    def do_eval() -> dict:
        row = {"step": tr.global_step, "exposures": round(tr.exposures, 4),
               "lr": tr.current_lr(),
               "train_ce_running": ("" if tr.running_ce is None
                                    else round(tr.running_ce, 6))}
        res = tr.evaluate()
        ceiling = bool(res.pop("_ceiling"))
        row.update({k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in res.items()})
        row["elapsed_s"] = round(time.time() - t0, 1)
        append_row(metrics, row, columns)
        tr.assert_frozen()
        torch.save(tr.state_dict(),
                   os.path.join(ckpt_dir, f"step_{tr.global_step:08d}.pt"))
        key = "top1" if tr.route == ROUTE_COMPREHENSION else "lex_exact"
        print(f"  [EVAL @ {row['exposures']} exp] lr={row['lr']} "
              f"{key}={row[key]} " +
              (f"top5={row['top5']} rank_med={row['rank_median']} "
               f"ce={row['retrieval_ce']}"
               if tr.route == ROUTE_COMPREHENSION else
               f"free={row['lex_exact_freear']} "
               f"pseudo={row['pseudo_exact']}/{row['pseudo_exact_freear']} "
               f"tf_tok={row['tf_token_acc']} ce={row['wm_ce']}"), flush=True)
        row["_ceiling"] = ceiling
        return row

    hit = False
    if tr.global_step in eval_steps:
        hit = do_eval()["_ceiling"] and not args.no_stop_at_ceiling
    while tr.global_step < max_steps and not hit:
        ce = tr.train_step()
        if args.log_every and tr.global_step % args.log_every == 0:
            print(f"[step {tr.global_step}/{max_steps}] exp={tr.exposures:.1f} "
                  f"lr={tr.current_lr()} ce={ce:.4f} run={tr.running_ce:.4f} "
                  f"| {time.time()-t0:.0f}s", flush=True)
        if tr.global_step in eval_steps:
            row = do_eval()
            if row["_ceiling"] and not args.no_stop_at_ceiling:
                print(f"[probe] CEILING REACHED at {row['exposures']} "
                      f"exposures; checkpoint saved; stopping.", flush=True)
                hit = True
    if tr.global_step not in eval_steps and not hit:
        do_eval()
    print(f"[probe] done: {args.route} {tr.widths} step={tr.global_step} "
          f"({tr.exposures:.1f} exp) -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
