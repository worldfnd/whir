# WHIR v1 with private masks

Over an odd field, the prover commits to $w\in\mathbb F^n$ and proves $\langle k,w\rangle=u$. The caller may choose $k$ after commitment. Each proof uses a fresh commitment. Self-derived construction implemented in `whir_prefix`. [Security argument](whir-zk-security-2026-11-04.md).

## 1. Put masks inside the available padding

Let $N=\operatorname{nextpow2}(n)$. The opening form has $N$ coefficients; entries after $n$ are ignored. Before each nonterminal sumcheck block, use $B=2^s$ columns, where $s$ is the number of folding rounds.

$$
b=\lceil n/B\rceil,\qquad D=\operatorname{nextpow2}(b+t),\qquad
F_j=r_j\Vert w_j\Vert0\in\mathbb F^D,\quad r_j\leftarrow\mathbb F^t.
$$

Split $w$ into consecutive blocks of at most $b$ entries; missing entries and columns are zero. The committed table is $F=F_0\Vert\cdots\Vert F_{B-1}$, of length $M=BD$. Insert zero weights for masks and padding into $k$, giving $A$ with $\langle A,F\rangle=u$.

For v1 `complete_age_check`, the first payload has $657900+80$ entries, including the outer sumcheck mask; the second has $655177$. With $B=32,t=256$, both give $D=32768$ and $M=2^{20}$. The prefixes fit without doubling the table.

## 2. Commit and add OOD claims

Write $U_c(Z)=\sum_j c_jZ^j$. RS-encode each column and Merkle-commit rows $(U_{F_0}(z),\ldots,U_{F_{B-1}}(z))$. Leaves include the oracle identifier and deterministic zeros for hash-block alignment; there are no leaf salts.

After the root, sample distinct OOD points outside the RS row domain and send $y_a=U_F(z_a)$. These evaluate the entire table, including prefixes. After the opening form and claimed value are fixed, receive batching scalars $\beta_a$ and update

$$
A\leftarrow A+\sum_a\beta_a(1,z_a,\ldots,z_a^{M-1}),\qquad
v\leftarrow u+\sum_a\beta_a y_a.
$$

We now prove $\langle A,F\rangle=v$. The first commitment's root and OOD answers precede the caller's choice of $k$.

## 3. Hide the sumcheck messages

Let $\ell=\log_2M$. Sample independent coefficients $c=(g_0,a_1,b_1,\ldots,a_s,b_s)$ and commit a separate prefixed RS table $T_g=r_g\Vert c\Vert0$. Define

$$
g(X)=g_0+\sum_{j=1}^s(a_jX_j+b_jX_j^2),\qquad
d=\sum_{x\in\{0,1\}^\ell}g(x).
$$

Send $d$, then receive $\rho$. Run $s$ quadratic sumcheck rounds on

$$
\sum_x\bigl[\rho\,\widetilde A(x)\widetilde F(x)+g(x)\bigr]=\rho v+d.
$$

Tildes denote multilinear extensions of coefficient tables. Each round sends the constant and quadratic coefficients before its challenge. For challenges $\alpha=(\alpha_1,\ldots,\alpha_s)$, the remaining claim is

$$
\langle\rho\operatorname{Fold}_\alpha(A),F'\rangle+\langle\eta,c\rangle=v',\qquad
F'=\operatorname{Fold}_\alpha(F),\quad
\eta=2^{\ell-s}(1,\alpha_1,\alpha_1^2,\ldots,\alpha_s,\alpha_s^2).
$$

The folded table includes the old prefixes. Neither $c$ nor its correction is disclosed.

## 4. Carry the private data through code-switch

Pack the next payload as $P=\operatorname{pad}(F'\Vert c\Vert0\Vert r_g)$. Zeros align the coefficient and prefix segments; the whole payload is padded to a power of two. Its initial weights are $\operatorname{pad}(\rho\operatorname{Fold}(A)\Vert\eta\Vert0)$.

Add fresh prefixes, commit the new table, and send its OOD answers. Only then sample rows from the old main and mask commitments. Their consistency equations are

$$
U_{F'}(z)=\langle\operatorname{eq}(\alpha),\mathrm{row}_F(z)\rangle,\qquad
U_{r_g}(\zeta)+\zeta^{t_g}U_c(\zeta)=\mathrm{row}_{T_g}(\zeta).
$$

The vector $\operatorname{eq}(\alpha)$ contains the public multilinear folding weights across the old columns. Each left side is a linear form on the new private payload; each right side comes from authenticated old rows. Add these equations and the new OOD equations using fresh batching scalars; keep the carried claim's coefficient one. This produces the next claim $\langle A_1,F_1\rangle=v_1$.

Repeat steps 3–4 while the payload exceeds the base size. Each code-switch must shrink the physical table.

## 5. Finish with a full random mask

At the base case, use one column. For $\langle A,F\rangle=v$, sample a full independent vector $h\in\mathbb F^D$. Commit $h$, send $b=\langle A,h\rangle$, then receive $\epsilon$ and reveal

$$
H=h+\epsilon F.
$$

Open both roots at the same fresh indices and check $U_H(z)=\mathrm{row}_h(z)+\epsilon\,\mathrm{row}_F(z)$. Use v1's ordinary sumcheck to check $\langle A,H\rangle=b+\epsilon v$. The outer verifier also checks complete proof consumption.

## What must remain private

The verifier sees roots, queried rows, OOD answers, masked sumcheck messages, $d,b$, and the terminal $H$. It never receives $c$, $r_g$, or a separate mask correction. A main prefix needs $t\ge q+h_{\rm OOD}$; a mask prefix needs $t_g\ge q_g$, counting all distinct queries during that commitment's lifetime. Terminal roots share query indices. These are joint field-disclosure budgets. Privacy of unsalted Merkle roots and paths remains an open proof obligation.
