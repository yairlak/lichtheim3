# Draft message for Yair

Hi Yair — update on the full-lexicon joint model, plus a question.

**We kept both 256 and 512 and ran the dorsal search you asked for.**
Ventral width matters enormously; dorsal width does not. From-scratch joint
1:2:3, 4 matched seeds:

- comprehension at u500: **enc/dec 256 → ~58%**, **512 → ~98%** (same
  direction in all 4 seeds); repetition unaffected by ventral width.
- dorsal WM probe 128/256/512 at matched exposure: lexical repetition
  ~99.99% at all three, pseudowords **.980 / .950 / .948** — so 128 is as
  good or better. We're keeping WM at 128.

**Late learning rate turned out to matter more than we expected.** Continuing
H512 to u750 left C ≈ 99.2% with everything fluctuating just under ceiling.
Dropping the LR 1e-4 → 3e-5 gave, in all 4 seeds: **naming exactly 100%** at
every checkpoint, fewer repetition errors, and LTM-only repetition
**.83 → .94** — with no cost to comprehension.

**Now the interesting bit — a clean tradeoff.** From that state we raised
*only* the comprehension LR back to 1e-4 (R and N stayed at 3e-5), to u1200:

| | C errors (of 27,981) | naming | repetition | LTM-only rep |
|---|---|---|---|---|
| all 3e-5 | 137.8 | 0 errors | 3.3 | **.935** |
| C-high | **123.8** | 0 errors | 4.5 | .870 |

C improves in **4/4 seeds** (−10, −14, −21, −11) but LTM-only repetition
drops ~6.5 pp. WM-only repetition stays at ceiling throughout, so the cost is
specifically in the ventral route — it looks like the encoder/semantic
surface moving faster than the ventral production pathway can stay
co-adapted, though we haven't proven that mechanism.

**Question — which direction would you prefer?**

1. **Extra repetition rehearsal (2:2:3)** from the C-high state, to test
   whether rehearsal restores LTM while keeping the C gain. This is our
   preferred option: N and C exposures stay matched, so repetition is the
   only factor.
2. **Your 1:2:4 proposal.** Happy to run it — but we'd suggest **1:2:4 vs
   2:2:4** as a pair, since 2:2:4 against the current 1:2:3 would change
   repetition *and* comprehension allocation at once and wouldn't be
   attributable.
3. **Keep tuning task-specific LRs** (a C LR between 3e-5 and 1e-4).

We're leaning towards 1. Nothing new is running while we decide.

Attached: exact recipe (populations, hashes, losses, samplers, the 1:2:3
update scheme, LR clock, evaluation definitions), per-seed plots, and the
results table. Note the ratio is Ueno-*inspired* experience allocation — the
optimizer and training loop are our modern implementation, not Ueno's.
