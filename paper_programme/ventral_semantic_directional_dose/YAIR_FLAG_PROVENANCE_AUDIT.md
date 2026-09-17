# YAIR_FLAG_PROVENANCE_AUDIT

Workstream: VENTRAL SEMANTIC DIRECTIONAL DOSE (design + audit pass). Written 2026-09-17. This is provenance archaeology; nothing was reconstructed by intuition.

**FLAG_PROPOSAL_STATUS=DOCUMENTED_SUBSTANTIVELY**

One contemporary source uses the word "flag" for a proposal attributed to Yair. It answers all three questions: the variable, where it enters, and the problem it addresses. It is not DOCUMENTED_EXACTLY because the source states that it is a paraphrase, not a transcript, and because the mechanism was explicitly left undefined.

---

## 1. Primary source

| field | value |
|---|---|
| path | `/Users/louishayot/Downloads/MEETING_NOTES_YAIR_EMMANUEL_2026-09-14.md` (outside the repository) |
| SHA256 | `c1983db3ee244182e3a3f11c25ed8500c36a01f5dd53d5b454b01d4a0032b70e` (verified in this pass) |
| size / mtime | 15,315 bytes · 2026-09-14 20:46 |
| read-only archival copy (byte-identical, outside git) | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/archives/ventral_directional_dose_design_20260917/MEETING_NOTES_YAIR_EMMANUEL_2026-09-14.md` |
| date / context | meeting of 14 Sept 2026: Yair Lakretz × Emmanuel Mandonnet × Louis Hayot |
| stated nature (header, lines 5–6) | "notes contemporaines de Louis rédigées après la réunion" [Louis's notes, written after the meeting]; "des **paraphrases / souvenirs de réunion**, pas des citations verbatim … ne doit pas être traité comme une transcription exacte" [paraphrases / recollections of the meeting, not verbatim quotes … must not be treated as an exact transcript] |
| prior in-repo citation | `paper_programme/gating_route_diagnostics/GATING_ROUTE_DIAGNOSTICS_RECAP.md:52–54`: "EXTERNAL INPUT: MEETING_NOTES_YAIR_EMMANUEL_2026-09-14.md · status: CONTEMPORARY_RELAYED_MEETING_NOTES (hypothesis source, not historical authority) · sha256: c1983db3…b70e", the same hash |

## 2. Recovered claim (candidate C1)

**Status: DOCUMENTED_SUBSTANTIVELY**

Section 3, "Proposition de Yair : dynamique d'attracteur sémantique" [Yair's proposal: semantic attractor dynamics], lines 47–93. Wording is quoted exactly from the notes, with an English gloss in brackets; the notes themselves are Louis's paraphrase.

* **Framing (line 49):** "Yair réfléchit à une modification possible du **gating / de la dynamique ventrale**." [Yair is considering a possible modification of the gating / of the ventral dynamics.]
* **Problem (line 53):** "La voie ventrale utilise actuellement une projection sémantique `ŝ` / `s_hat`, qui n'est pas nécessairement suffisamment précise." [The ventral route currently uses a semantic projection ŝ that is not necessarily precise enough.]
* **Idea (line 55):** "introduire une **boucle sémantique auto-amplificatrice / attractor-like**, avec une logique rappelant le copy-back de Ueno et al." [introduce a self-amplifying / attractor-like semantic loop, recalling Ueno et al.'s copy-back.]
* **Loop (lines 59–64, "Idée paraphrasée" [paraphrased idea]):**
  1. "la voie ventrale produit une première représentation sémantique `ŝ`" [the ventral route produces a first semantic representation ŝ];
  2. "cette représentation est réinjectée / utilisée pour revenir vers l'espace GloVe ou un mécanisme de récupération sémantique" [it is fed back / used to return to GloVe space or a semantic retrieval mechanism];
  3. "on obtient une représentation sémantique raffinée" [a refined semantic representation is obtained];
  4. "cette représentation raffinée modifie l'état ventral / `z_LTM`" [the refined representation modifies the ventral state / z_LTM];
  5. "une nouvelle estimation `ŝ` est produite" [a new estimate ŝ is produced];
  6. "la boucle peut raffiner progressivement le concept abstrait activé" [the loop can progressively refine the activated abstract concept].
* **Target dynamics (line 66):** "l'activité sémantique se rapproche progressivement d'un concept stable / plus précis" [semantic activity progressively approaches a stable / more precise concept].
* **The flag itself (lines 70–77):** "Ajouter un flag du type : `semantic_attractor = True / False` · `False` : comportement actuel ; `True` : boucle sémantique / refinement récurrent." [Add a flag such as `semantic_attractor = True / False`; False = current behaviour; True = semantic loop / recurrent refinement.]
* **Status (line 81):** "**PROPOSITION À TESTER**, pas décision finale d'architecture." [A proposal to test, not a final architecture decision.]
* **Explicitly unspecified (lines 83–93):** the exact definition of the loop, what is fed back, GloVe as retrieval bank or continuous target, the number of iterations, whether gradients cross the loop, the impact on the gate, and the impact on Naming, Comprehension, ventral-only repetition and stability.

### Answers to CENTRAL's three questions

| question | answer | status |
|---|---|---|
| 1. What variable did the flag represent? | A boolean configuration switch, `semantic_attractor ∈ {True, False}`, enabling or disabling a recurrent semantic refinement loop acting on ŝ. False = current model. | DOCUMENTED_SUBSTANTIVELY |
| 2. Where did it enter the model? | The ventral route. ŝ is fed back through GloVe space or semantic retrieval into the ventral state `z_LTM`, producing a new ŝ, iteratively. It is framed as a change to the "gating / ventral dynamics". The exact wiring is **not specified**. | DOCUMENTED_SUBSTANTIVELY (wiring NOT_RECOVERED) |
| 3. What problem was it meant to solve? | The ventral semantic projection ŝ "not necessarily precise enough"; attractor-like settling toward a stable, more precise concept. | DOCUMENTED_SUBSTANTIVELY |

**Relation to the current workstream (observation, not an equivalence claim).**
* Proposal C1 targets the same quantity (ŝ) and the same direction of change (toward GloVe / retrieved lexical semantics) that the closed ventral-interface diagnostic localized and the directional-dose design probes.
* The closed diagnostic concluded that a recurrent semantic attractor is **not established as necessary**.
* CENTRAL's instructions list "DO NOT IMPLEMENT A SEMANTIC ATTRACTOR" and "DO NOT IMPLEMENT THE 'FLAG'" as separate items. **No local source documents a flag distinct from `semantic_attractor`.** Whether CENTRAL means a different concept is unresolved (§6).

## 3. Other candidates (NOT identified as the flag)

| id | concept | source | wording | status |
|---|---|---|---|---|
| C2 | Fixed symmetric fusion baseline `gate = 0.5 / 0.5` | same notes, §8 (lines 243–264) | "Une baseline explicitement demandée / jugée importante : gate = 0.5 / 0.5 … fusion fixe et symétrique des voies, sans routage dynamique dépendant de la confiance." [A baseline explicitly requested / judged important: gate = 0.5 / 0.5, a fixed symmetric fusion of the routes without confidence-dependent routing.] No speaker named; not called a flag. | POSSIBLE_BUT_NOT_IDENTIFIED_AS_FLAG |
| C3 | Constraints on g (α, τ or gate formula; avoid saturation) | same notes, §9 (lines 266–291) | "le gate ne devrait peut-être pas saturer trop facilement vers une route … contraintes supplémentaires … sur alpha ; tau / threshold ; ou sur la formulation même du gate" [the gate perhaps should not saturate too easily toward one route … further constraints on alpha, tau / threshold, or the gate formula itself]. No speaker named; not called a flag. | POSSIBLE_BUT_NOT_IDENTIFIED_AS_FLAG |
| C4 | Gate threshold as a hyperparameter | `~/Desktop/ENS LSCP/prompt_archi-notesYair.docx` (10 July 2026; Louis relaying Yair), as reported by the search agent; not re-read in this pass | "0.5 devrait être un hyperparamètre … c_0 ou t" [0.5 should be a hyperparameter, c_0 or t]. Implemented as configurable `gate_threshold` in commit `8b5d47856a9e2df869a7c314ced8d351e09e98da` (2026-07-16). Not called a flag. | POSSIBLE_BUT_NOT_IDENTIFIED_AS_FLAG |
| C5 | Bilateral gate `g = f(c_LTM, c_WM)` | `~/Desktop/LICHTHEIM3_FINAL_MODEL_CONVERGENCE_PLAN_2026-09-16.md` (Family G), as reported by the search agent | Not attributed to Yair; not called a flag. | POSSIBLE_BUT_NOT_IDENTIFIED_AS_FLAG |
| — | task identity; word/pseudoword lexicality indicator; tied lesion mask | — | No source calls any of these Yair's flag. | NOT_RECOVERED |

## 4. EXISTING_GATE (documented separately; **not** YAIR_FLAG)

Live code at `0f25b5b87efd8038dfc4c86b5ca2935bf7b178c7`; model code identical to the V6 code commit.

```
c_LTM   = max_j  ( normalize(ŝ) · normalize(bank_raw)_j )                  models/ltm_route.py:182–185 (lexical_field)
g       = sigmoid( alpha · (c_LTM − gate_threshold) )                        models/gating.py:49–51  (word-level, expanded over steps; no parameters)
fusion  = premotor = g · ltm_premotor + (1 − g) · wm_premotor                models/gating.py:56;  logits = motor(premotor)  models/dual_route.py:85–88
L_gate  = (mean g − usage_prior)²,  weight 0.05                             losses.py:53–55,66
V6 values: alpha 2.0, gate_threshold 0.7, usage_prior 0.5  (checkpoint config; code defaults 4.0 / 0.5 / 0.5)
```

* **Documented purpose:** `models/gating.py:1–15`, "The gate: error-suppression (lexicality routing). A confident lexical match exerts top-down suppression on the WM buffer … Real words land close to a known lexeme → high confidence → the ventral route wins; non-words land far from every lexeme → low confidence → the dorsal buffer has to carry the trial."
* **Origin:** the file is present in the root commit `56b665a5b2f9c513235d1f38b3215da4bdc9823b` (2026-06-29, author yairlak). The threshold was made configurable in `8b5d478…` (2026-07-16).
* **Gradients:** see `VENTRAL_INTERFACE_TRAINING_SUPERVISION_AUDIT.md` §F.
* **No source identifies the existing gate with the flag proposal.** C1 is framed as a change to "gating / ventral dynamics", and `paper_programme/gating_route_diagnostics/CODE_AUDIT_GATE.md:92` notes that "a semantic-attractor loop makes `ltm` depend on `g`". Neither statement establishes an equivalence.

## 5. What was searched

Search was delegated to a read-only search agent. The primary source and the prior in-repo citation were re-verified directly in this pass (hash, lines 1–100 and 236–300, section index).

* `CHAT_ARCHAEOLOGY/`: all 9 files. Only "flagged as problem" usages. The chat record ends 2026-09-12, before the meeting.
* `meeting_yair_inputs/` and `lichtheim3_yair_meeting_inputs.zip`: contents listed. Metrics and README only; no notes.
* `stage_source_of_truth/`: phase audits, reader packs, targeted-archaeology and supervisor-chat provenance, including the full Discord evidence archive (June to 7 Sept). About 104 flag/indicator/gate hits reviewed; all technical (CLI, preservation, small-cell, enablement flags).
* All worktrees and the repository root: about 2,655 `.md/.txt/.tex/.yaml` files plus `.py/.tsv/.json/.sh`; 156 unique "flag" lines. No model proposal except the in-repo citation of the meeting notes and contracts forbidding `semantic_attractor` work (`gating_route_diagnostics/EXPERIMENT_CONTRACT.md`, `GATE_X_LESION_EXPERIMENT_CONTRACT.md`).
* Git (`lichtheim3/.git`, 196 commits, all branches): messages and bodies searched for flag, yair, emmanuel; history of `models/gating.py`.
* Secondary, read-only: local Claude transcripts and memory. No pre-09-16 user mention of a Yair flag; the 09-16 CENTRAL prompt is not evidence. Desktop and Downloads Markdown/Word documents since June: the source was found in Downloads.
* **Excluded:** site-packages, `.git` internals, checkpoints, binaries, lexicon TSVs, and Gmail/Drive/Calendar (connectors not authorized in this session).
* **Search terms:** flag, drapeau, task/condition/context/route flag, indicator, binary indicator, one-hot, cue, lexicality, familiarity, word/pseudoword, nonword, attractor, Yair, Emmanuel, meeting, semantic, naming, comprehension, repetition, gate.

## 6. Unresolved

1. **Loop semantics were never defined:** what is fed back, retrieval vs continuous target, iterations, gradient flow, interaction with g.
2. **No verbatim wording** from Yair or Emmanuel. The only source is Louis's post-meeting paraphrase, and the Discord archive ends 7 Sept.
3. **The meeting notes are not in the repository** or the Source-of-Truth. Only this pass's read-only archival copy and a hash citation exist.
4. **CENTRAL's separate "attractor" and "flag" prohibitions.** Whether CENTRAL refers to a concept other than `semantic_attractor = True/False` cannot be established locally. If so, its source is NOT_RECOVERED.
5. **Unsearched sources:** Gmail, Google Drive and Calendar connectors are unauthorized in this session.
