# Next-experiment decision memo (no experiment implemented)

## The state we are choosing from

At u1200 there are **two** live states per seed, and picking the branch point
is itself a decision:

| | C errors | N errors | R errors | LTM-only rep |
|---|---|---|---|---|
| all-3e-5 | 137.75 | 0 | 3.25 | .9349 |
| C-high | 123.75 | 0 | 4.50 | .8695 |

C-high is better on C in 4/4 seeds and worse on LTM in the mean. Neither is
dominant, so a "best current model" does not exist yet.

## Exposure arithmetic (this decides the design)

Per macro-cycle, with |R| = |N| = 29,571 (463 batches/pass) and |C| = 27,981
(438 batches/pass):

| ratio | steps/cycle | R | N | C | exposures per 1000 cycles (R / N / C) |
|---|---|---|---|---|---|
| 1:2:3 | 6 | 1 | 2 | 3 | 2.16 / 4.32 / 6.85 |
| 2:2:3 | 7 | 2 | 2 | 3 | 4.32 / 4.32 / 6.85 |
| 1:2:4 | 7 | 1 | 2 | 4 | 2.16 / 4.32 / 9.13 |
| 2:2:4 | 8 | 2 | 2 | 4 | 4.32 / 4.32 / 9.13 |

**The key structural facts:**

- **1:2:3 → 2:2:3** leaves N and C *per cycle* unchanged. At the same cycle
  count both are matched exactly, R doubles, and it costs 1.167× the steps.
  **Clean one-factor test of extra repetition.**
- **1:2:4 → 2:2:4** likewise: N and C matched, R doubles, 1.143× steps.
  **Clean one-factor test of extra repetition under a C-heavy regime.**
- **1:2:3 → 1:2:4** leaves R and N per cycle unchanged; C rises 4/3.
  **Clean one-factor test of extra comprehension.**
- **1:2:3 → 2:2:4 moves BOTH R and C.** Not interpretable as a single factor.

## The options

### A. Continue 1:2:3 with task-specific LR tuning
*Hypothesis:* the C/LTM tradeoff is an LR-operating-point problem; some C LR
between 3e-5 and 1e-4 keeps most of the C gain at less LTM cost.
*Factor:* comprehension LR (continuous).
*Confounds:* few — the machinery is proven and the branch is paired. But
"tune a scalar" has no natural stopping point and risks becoming a sweep,
which is explicitly out of scope.
*Information gain:* **Low–moderate.** Likely maps a tradeoff curve rather
than removing the tradeoff. Tells us little mechanistically.

### B. 2:2:3 repetition rescue
*Hypothesis:* the LTM cost of faster C is a **rehearsal deficit** — the
ventral production pathway falls behind a faster-moving encoder — so
doubling repetition presentations restores LTM while keeping the C gain.
*Factor:* repetition allocation only (N and C matched exactly).
*Confounds:* the branch point choice (from C-high, where the deficit exists,
is the informative one); +16.7% steps for matched N/C, so compare at matched
exposures, never at matched step. Historical precedent is encouraging but was
obtained at H128 under a different LR.
*Information gain:* **High.** It directly tests the mechanism proposed in
interpretation I3, and it is the *only* option that can dissolve the tradeoff
rather than trade along it. Also the historically motivated Ueno-adjacent
manipulation.

### C. Yair's 1:2:4
*Hypothesis:* comprehension is simply under-allocated experience; more C
presentations resolve the residual faster than a higher C LR, and without the
LR's destabilising effect on shared weights.
*Factor:* comprehension allocation only (R and N matched).
*Confounds:* raising C experience may cause the *same* ventral interference
as raising the C LR — in which case it reproduces the tradeoff by another
route. That is itself informative (it would distinguish "amount of C signal"
from "size of C step"), but it is not a rescue.
*Information gain:* **Moderate–high**, and it directly addresses Yair's own
proposal, which has standing value for the collaboration.

### D. 2:2:4
*Hypothesis:* combined — more C experience *and* enough repetition rehearsal
to protect LTM.
*Factor:* relative to the current 1:2:3, **two factors move at once**
(R doubles, C rises 4/3). Relative to 1:2:4 it is a clean one-factor R test.
*Confounds:* as a next step from 1:2:3 it is uninterpretable. It only becomes
clean as the second half of a 1:2:4 vs 2:2:4 pair.
*Information gain:* **High but only in the right pairing.** Run alone against
the current state, it would tell us a combination works without telling us
which component did the work.

## Recommendation

**B (2:2:3), branched from the C-high u1200 state, paired against a C-high
1:2:3 continuation** — matched on N and C exposures, not on steps.

Reasoning: C-high is the state that *exhibits* the deficit, so it is where a
rehearsal rescue is testable; the comparison is one-factor by construction;
and it is the only candidate that could remove the tradeoff instead of
sliding along it. If it works, we get C-high's C **and** all-3e-5's LTM. If
it fails, the "rehearsal deficit" reading of I3 is falsified, which is worth
knowing before any further LR tuning.

**If Yair prefers his 1:2:4 direction**, the scientifically correct form is a
**1:2:4 vs 2:2:4 pair** (matched N/C, R the only factor) rather than 1:2:4 or
2:2:4 alone against the current 1:2:3. That answers the extra-R question
inside the C-heavy regime he is interested in, and costs the same two arms.

Deferred regardless: margin/hard-negatives (trigger still unmet), RN|C banks
(no interference signature), architecture changes, loss changes.

**Prerequisite before any of this:** run the u1200 per-item C audit on both
arms. If the residual has become small and stable, the whole ratio question
is moot and the margin trigger may finally be live.
