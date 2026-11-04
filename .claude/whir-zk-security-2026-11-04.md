# Security argument for private-mask WHIR

Model: one fresh commitment, one adaptive linear opening, odd field, classical programmable random oracles. Concrete hashes are modeled as oracles. [Protocol](whir-zk-construction-2026-11-04.md); [exact arithmetic checker](whir-zk-accounting-2026-11-04.py).

Current `v2-unsalted` rows have no independent salts. The Merkle privacy argument below applies only to the earlier salted layout; the complete security argument must be rechecked against the current encoding.

## Definitions and claims

An oracle has $B$ data columns, degree bound $<D$, and $m$ rows. Distance counts unequal complete data rows; salts and identifiers belong to the commitment layer. Its code has distance at least $m-D+1$. Lists contain every coefficient tuple within the selected row radius, including arbitrary malicious padding.

Evaluation binding means that the commitment-stage root and OOD answers identify a payload $w$ before the opening form $k$ is chosen, and an accepted opening satisfies $u=\langle k,w\rangle$, except with the stated probability. It does not assert knowledge of randomness reproducing an arbitrary malicious root.

Privacy means an online simulator can produce the entire view from public parameters and authorized opening outputs. It receives $u$ when the verifier chooses $k$, after commitment. Commitment reuse and additional openings are outside this budget.

## Completeness

Prefix insertion is linear and inserts zero weights into covectors. Quadratic sumcheck preserves the endpoint sum. After $s$ folds,

$$
v'=\rho\langle\operatorname{Fold}(A),\operatorname{Fold}(F)\rangle
+2^{\ell-s}\left(g_0+\sum_j(a_j\alpha_j+b_j\alpha_j^2)\right).
$$

Packing places these coefficients at the positions weighted by the next carry. Main-row and mask-row equations are identities for the encoded coefficients. The terminal identity follows from $H=h+\epsilon F$. Challenges zero and one need no division; only the odd-characteristic assumption is required.

## Joint field-view privacy

**Prefix rank.** For $q$ distinct queried row points, let $V(Z)=\prod_b(Z-x_b)$. Any required row differences can be met by interpolating a prefix polynomial of degree $<q$ in each column. Its remaining freedom is $V(Z)S(Z)$. The $h$ whole-table OOD equations can all be corrected through column zero: $V(z_a)\ne0$, the OOD points are distinct, and $\deg S<h$. Therefore $t\ge q+h$ makes the joint row/OOD map surjective.

Given any hidden payload, fresh prefixes remain uniform subject to their already-sent OOD equations. Payload constraints have zero weights on these fresh prefixes; only the OOD equations constrain them. Later row replies are consequently uniform, conditioned on those OOD answers and the preceding public history.

**Sumcheck masks.** The mask total $d$ is uniform. Condition on $d$, the hidden table, and previous round messages. For the current pair $(a_j,b_j)$, the two transmitted coefficients have linear part

$$
\begin{pmatrix}
-\sigma/2&-\sigma/2\\
0&\sigma
\end{pmatrix},
\qquad \det=-\sigma^2/2\ne0,
$$

where $\sigma$ is the current nonzero power of two. Thus each pair of messages is uniform. After all messages, every coefficient of $G$ may be determined; its independent prefix still hides its $q_g\le t_g$ oracle evaluations.

**Recursion.** The next payload contains the folded table, $G$, and its encoding prefix. Earlier disclosures restrict that payload, not its new independent prefixes. Applying the same rank argument at each root proves the conditional response laws over the entire history.

**Terminal.** On a supported history $\langle A,F\rangle=v$. If $A\ne0$, $b=\langle A,h\rangle$ is uniform, and $H$ is uniform on

$$
\{H:\langle A,H\rangle=b+\epsilon v\},
$$

independently of $F$. If $A=0$, supported histories have $v=0$, so $b=0$ and $H$ is unrestricted uniform. The final main rows retain their conditional uniform law; mask rows are determined by $U_h=U_H-\epsilon U_F$ at the same indices. Sumcheck on public $H,A$ adds no secret disclosure.

These laws give an efficient online simulator: sample OOD values, mask totals, masked round pairs, and oracle replies; update the public claims; sample terminal $H$ by solving one nonzero coordinate of $A$; then run ordinary sumcheck on public data. Each simulator kernel equals the honest conditional kernel on supported histories. Induction keeps every generated history supported, establishing $A=0\Rightarrow v=0$.

They also give a transcript-preserving coin bijection between witnesses with the same authorized outputs: translate main prefixes to preserve rows and OOD values, translate all $G$ coefficients to preserve its messages, translate its prefix to preserve its rows, and recurse. Set $\Delta h=-\epsilon\Delta F$ at termination. Fixed linear right inverses make the negative shift the inverse. The unchanged transcript makes adaptive verifier choices identical, including zero $\rho$ or $\epsilon$.

## Soundness in the binding-oracle model

Let $\Lambda_i$ be the row-close tuple list at main root $i$, with $|\Lambda_i|\le L_i$. A state is good if some $f\in\Lambda_i$ satisfies its carried claim $\langle A_i,f\rangle=v_i$. A missing list or a false claim for every candidate is bad. The initial external/OOD batching turns a bad collection of individual claims into a good combined claim with probability at most $L_0/Q$.

**Folding.** [BCHKS, Theorem 4.6, pages 28–29](https://www.math.toronto.edu/swastik/rs-proximity-gaps-2025.pdf) bounds the number of exceptional scalar combinations below the Johnson radius. Represent remaining column tuples by an extension field $E/\mathbb F$: degree-bounded $E$ polynomials are exactly tuples of degree-bounded base-field polynomials, and scalar agreement is complete-row agreement. A bound of $H_i$ bad seeds in $E$ also bounds bad seeds in $\mathbb F$ by $H_i$, giving failure $H_i/Q$.

Write each high-bit fold as $A+c(B-A)$. Outside the exceptional seeds, every close folded polynomial lifts to a close input tuple. Their agreement on more than $D_i-1$ rows forces the exact folded polynomial identity. Repeating conditionally through $s$ challenges costs $sH_i/Q$.

**Masking and code-switch.** Fix the old main and mask lists before $\rho$. For each pair $(f,G)$ with a false old claim, at most one $\rho$ repairs the announced masked sum. Partial sumcheck costs $2s/Q$ for that pair. Unioning the fixed lists costs $(1+2s)L_iL_g/Q$, even if the prover selects its pair later.

Each next-root candidate is fixed before old-row queries. Its payload defines the predicted main polynomial and the predicted mask polynomial. The latter includes the carried prefix and designated mask coefficients, with zeros elsewhere. A close prediction belongs to the old list; arbitrary malicious padding is not assumed zero. A far prediction passes all $q$ sampled locations with probability at most $(1-\delta)^q$. Sampling with replacement and deduplicating afterward preserves this no-hit probability.

If any carry, row, or new OOD equation is false for a fixed next candidate, fresh batching cancels its error with probability at most $1/Q$. Prefix insertion applies to payload weights; whole-table OOD weights cover the physical table. Therefore

$$
\epsilon_i\le
\frac{sH_i+(1+2s)L_iL_g+L_{i+1}}{Q}
+L_{i+1}\left[(1-\delta_i)^{q_i}+(1-\delta_g)^{q_g}\right].
$$

**Terminal.** Both roots and $b$ precede $\epsilon$; $H$ precedes the queries. Apply correlated agreement directly to the fixed pair of root arrays. Outside $H_{\rm final}/Q$, any close $H$ is exactly $g+\epsilon f$ for a jointly close pair. If the old claim is false, each fixed pair satisfies its terminal inner-product claim for at most one $\epsilon$. A far $H$ misses the shared queries with probability at most $(1-\delta_{\rm final})^q$. Public full sumcheck adds $2\log_2D_{\rm final}/Q$:

$$
\epsilon_{\rm final}\le
\frac{H_{\rm final}+L_{\rm final}L_h+2\log_2D_{\rm final}}{Q}
+(1-\delta_{\rm final})^q.
$$

Every accepted path from a bad initial state crosses a bad-to-good switch or accepts a bad terminal state. Sum these bounds and the initial batching loss.

**Before-form identification.** The initial list is fixed before its OOD challenge. Distinct tuples give distinct whole-table polynomials of degree $<M_0=B_0D_0$. With one OOD point outside the row domain, collisions cost at most

$$
\epsilon_{\rm identify}\le
\binom{L_0}{2}\frac{M_0-1}{Q-m_0}.
$$

Outside that event, at most one candidate matches the commitment-stage answer; select it before $k$. If none matches, successful opening requires an already-counted batching/proximity failure. This selection argument is separate from the initial $L_0/Q$ batching loss.

## Concrete interactive parameters

Use BN254 scalar modulus $Q$, $N=2^{20}$, $s=5$, base payload $2048$, one OOD answer per main root, and prefixes of length $256$. The main $(B,D,m)$ schedule is

$$
(32,65536,262144),\quad(32,8192,32768),\quad
(32,1024,4096),\quad(1,4096,16384).
$$

Each mask has $(D_g,m_g)=(512,2048)$. There are three switches and eight roots including the terminal mask. Both regimes use the same code and proof; only the analysis and query count differ.

- Unique: radius $\delta=3/8$, list bound $L=1$, $q=195$.
- Johnson/list: radius $\delta=19/40$, list bound $L=40$, $q=148$.

For the theorem's reduced rate $r=(D-1)/m$,

$$
r\ge999/4000,\qquad \sqrt r\ge1999/4000,\qquad r<1/4.
$$

Two unique-radius candidates would differ on at most $2\delta m=3m/4$ rows, below the code's distance. For list decoding, count candidates matching each received row. Distinct tuples share at most $D-1$ rows; pair counting and the sum-of-squares bound give

$$
L\le\frac{1-r}{(21/40)^2-r}\le\frac{1200}{41}<40.
$$

Theorem 4.6 uses

$$
\mu=\max\!\left\{\left\lceil\frac{\sqrt r}{1-\sqrt r-\delta}\right\rceil,3\right\},\qquad
H=\frac{2(\mu+1/2)^5+3(\mu+1/2)\delta r}{3r^{3/2}}m+\frac{\mu+1/2}{\sqrt r}.
$$

Here $\mu\le4$ for unique decoding and $\mu\le20$ for list decoding. The displayed rational rate bounds give the checker's integer upper bounds on $H_i$. Including every round and initial OOD identification, before sampler bias,

$$
E_{\rm unique}(q)=\frac{14902126304}{Q}+7(5/8)^q,
$$

$$
E_{\rm list}(q)=\frac{29228498079722}{Q}
+241(21/40)^q
+\frac{780(2^{21}-1)}{Q-262144}.
$$

The exact checker also adds the BN254 challenge-reduction bias. Each ordinary field draw costs at most $Q/2^{512}$ in statistical distance; an accepted OOD draw costs at most four times that because its allowed set has probability above $1/2$. There are at most $51+6q$ such units in this profile. Exact rational comparisons give $E_{\rm unique}(195)<2^{-129}$ and $E_{\rm list}(148)<2^{-129}$. These are global interactive bounds, not per-stream or unconditional NIZK guarantees.

For unpadded payloads $2^{19}<n\le2^{20}-8192$, the 32 prefixes fit inside $M_0=2^{20}$. Main column lengths are $32768,4096,512,2048$; row counts remain four times those lengths. Use $r\ge511/2048$ and $\sqrt r\ge999/2000$. The same calculation gives, before sampler bias,

$$
E_{\rm unique}(q)=\frac{7461911770}{Q}+7(5/8)^q,
$$

$$
E_{\rm list}(q)=\frac{14635526956460}{Q}+241(21/40)^q
+\frac{780(2^{20}-1)}{Q-131072}.
$$

Sampler bias uses $50+6q$ units because the terminal table has 11 variables. The same query counts retain error below $2^{-129}$. Payload/form embedding maps exactly the first $n$ indices by $i\mapsto\lfloor i/\lceil n/32\rceil\rfloor D+t+(i\bmod\lceil n/32\rceil)$; trailing form weights are ignored, matching a zero-extended witness. The joint-privacy and binding arguments above are unchanged.

## Merkle and Fiat–Shamir compilation

Model SHA-256 Merkle hashing $H_M$, SHA3-256 form hashing $H_F$, and SHAKE128 absorbed-prefix streams $H_X$ as separate classical random oracles. Instantiating these oracles with the fixed hashes is a heuristic, not a proved replacement theorem. Private coins are uniform in the proof; actual private randomness adds its PRG distinguishing advantage.

The pinned transcript adapter appends absorbed bytes and resets its XOF reader; consecutive squeezes consume one prefix's stream. Count a complete absorbed-prefix stream once, including query indices and the following batching scalars. Do not count each squeeze as independent randomness.

**Causal binding.** At a root's first relevant prefix query, snapshot its recorded hash preimages. Complete unavailable or invalid field rows with zero vectors. Any later inconsistent accepted opening requires a new preimage for a missing label or a collision. With at most $R$ roots per proof, missing target labels number at most $2T_M+RT_X$. A form digest fixes a recorded canonical preimage before the opening prefix; later supplying a preimage incurs a digest-guess or collision event. Interpret form encodings under the fixed field, dimensions, and representation in that prefix's configuration.

Here $T_M,T_F$ count all relevant hash calls, including preprocessing, honest computation, and verification. $T_X$ counts distinct absorbed-prefix streams queried by the adversary or verifier, including speculative queries; extensions of one stream count once. Conservative bounds are

$$
\epsilon_M\le\frac{(3T_M+RT_X+1)^2}{2^{256}},
\qquad
\epsilon_F\le\frac{(T_F+T_X+1)^2}{2^{256}}.
$$

Outside these events, candidate arrays and the form are fixed before each fresh stream node. A node's field events have the interactive bounds above; query-dependent batching consumes fresh later bytes in that same stream. Every accepted false-opening path contains a bad node. Rewinding revisits the same stream. A conservative state-restoration union bound is

$$
\Pr[\mathrm{false\ opening}]\le
\min\{1,\ T_XE_{\rm interactive}+\epsilon_M+\epsilon_F\},
\qquad R=8\ \text{for the reviewed profile}.
$$

Authenticated row/path hints need not be absorbed: off the causal-binding event, the root and queried positions determine their data. The argument does not treat arbitrary unbound hints as empty ordinary messages.

**Merkle privacy of the previous salted layout (inapplicable to current code).** Sample independent random labels for the entire tree at commitment, program internal child-pair hashes, and defer salted leaf preimages until a row is opened. Cache repeat openings. Internal inputs are exactly 64 bytes; leaf inputs exceed 64 bytes. Leaves have an oracle identifier but no explicit index/session nonce, so the guessing bound includes all leaves.

Let $V$ be the total assigned node labels, $L$ the total leaves, and $a\ge2$ the minimum independent field salts per leaf, across all sessions. A single lazy-oracle coupling bounds node-label collisions, guesses of unexposed child labels, guesses of unopened salted leaves, and duplicate leaf preimages:

$$
\Delta_{\rm ZK}\le\min\!\left\{1,\
\frac{V(V-1)/2+2VT_M}{2^{256}}
+\frac{LT_M+L(L-1)/2}{Q^a}
+\operatorname{Adv}_{\rm PRG}\right\}.
$$

Legitimate verification of exposed paths is not a guessing abort. Keep the same $H_X$ and $H_F$ oracles in both privacy worlds; evaluate their outputs normally. The transcript-preserving coin translation recomputes the same shifts in its inverse, so it also preserves every adaptive XOF query and answer. No XOF programming or prequery term is needed.

The source remains experimental pending team review.
