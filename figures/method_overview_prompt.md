# README method figure

The checked-in asset is `assets/quotientopt_method_overview.png` (1933 × 813 PNG). It was drawn with image generation using the ManifoldFlow method overview as a layout and visual-style reference. `draw_method_overview.py` remains an editable, code-drawn schematic variant; it does not produce the checked-in asset.

## Visual specification

Use an ultra-wide, white-background academic method diagram with the same dense five-panel upper flow and three-panel lower explanation as the ManifoldFlow overview. Use fine saturated blue, green, orange, and purple rounded panel outlines; faint tinted interiors; short arrows between modules; serif mathematics; compact bold sans-serif headings; rectangular matrix tiles with visible cells; tight alignment and minimal blank space. No outer title, step numbers, logos, or decorative illustrations.

## Exact content

Upper row, left to right:

1. **Factorization (current factors):** tall blue $B_t$ times wide blue $A_t$ equals purple $X_t$. $X_t=B_tA_t$. Label dimensions $m\times r$, $r\times n$, and $m\times n$.
2. **Product-preserving balance:** blue $(B_t,A_t)$ maps to green $(\bar B_t,\bar A_t)$ through QR and a small SVD, and yields the same purple $X_t$. $\bar B_t\bar A_t=X_t$ and $\bar B_t^\top\bar B_t=\bar A_t\bar A_t^\top=H_t$.
3. **Chain-rule gradient:** orange $G_t$ branches into $g_B=G_t\bar A_t^\top$ and $g_A=\bar B_t^\top G_t$. Show $\bar B_t^\top g_B=g_A\bar A_t^\top$.
4. **Transport and horizontal memory:** two compact inner columns for transport/EMA and horizontal projection. $U=\beta\widetilde M+(1-\beta)g$. The $r\times r$ Sylvester variable $\Omega$ solves $(H+\varepsilon I)\Omega+\Omega(H+\varepsilon I)=\bar B^\top U_B-U_A\bar A^\top$. Then $M_B=U_B-\bar B\Omega$ and $M_A=U_A+\Omega\bar A$.
5. **Factor step and recompose:** blue $B_{t+1}$ times blue $A_{t+1}$ equals purple $X_{t+1}$. $B_{t+1}=\bar B_t-\eta M_B$, $A_{t+1}=\bar A_t-\eta M_A$, and $X_{t+1}=B_{t+1}A_{t+1}$.

Lower row:

1. **Gauge symmetry and balancing:** two visually distinct factor pairs produce one matrix. $(B,A)\sim(BQ,Q^{-1}A)$, with $(BQ)(Q^{-1}A)=BA$.
2. **State transport before projection:** past state $M_{t-1}$ maps to transported $\widetilde M_{t-1}$, then raw $U$, then $\Omega$. $\widetilde M_B=M_{B,t-1}Q$ and $\widetilde M_A=Q^{-1}M_{A,t-1}$.
3. **Horizontal update:** raw $U$ separates into projected $M$ and gauge motion. Show the regularized residual $\bar B^\top M_B-M_A\bar A^\top=2\varepsilon\Omega$ and a purple $\Omega$ tile. Exact horizontality follows when $\varepsilon=0$.

The style reference supplies only composition and visual language. All formulas and method content above belong to QuotientOpt. Proofread the generated image against this specification before publication.
