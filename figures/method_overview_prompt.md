# How the README figure was made

The README displays `assets/quotientopt_method_overview.png` (1933 × 813 pixels). The image was generated using the ManifoldFlow method figure as a layout and style reference. `draw_method_overview.py` draws a simpler, editable version; it does not recreate the README image.

## Layout and style

Use a wide white canvas. Put five connected steps across the top and three explanation panels below. Keep the panels close together, as in the ManifoldFlow figure. Use thin blue, green, orange, and purple borders, lightly tinted backgrounds, short arrows, bold headings, and readable equations. Draw each matrix as a grid of colored cells. Leave out an overall title, step numbers, logos, and decoration.

## Text and equations

Upper row, left to right:

1. **Factorization (current factors):** draw blue $B_t$ times blue $A_t$ equals purple $X_t$. Show $X_t=B_tA_t$ and the sizes $m\times r$, $r\times n$, and $m\times n$.
2. **Product-preserving balance:** show blue $(B_t,A_t)$ becoming green $(\bar B_t,\bar A_t)$ through QR and a small SVD. The product stays the same: $\bar B_t\bar A_t=X_t$. Also show $\bar B_t^\top\bar B_t=\bar A_t\bar A_t^\top=H_t$.
3. **Chain-rule gradient:** show orange $G_t$ branching into $g_B=G_t\bar A_t^\top$ and $g_A=\bar B_t^\top G_t$. Add $\bar B_t^\top g_B=g_A\bar A_t^\top$.
4. **Transport and horizontal memory:** split the panel into “Transport + EMA” and “Horizontal projection.” Show $U=\beta\widetilde M+(1-\beta)g$. The small $r\times r$ matrix $\Omega$ solves $(H+\varepsilon I)\Omega+\Omega(H+\varepsilon I)=\bar B^\top U_B-U_A\bar A^\top$. Then show $M_B=U_B-\bar B\Omega$ and $M_A=U_A+\Omega\bar A$.
5. **Factor step and recompose:** draw blue $B_{t+1}$ times blue $A_{t+1}$ equals purple $X_{t+1}$. Show $B_{t+1}=\bar B_t-\eta M_B$, $A_{t+1}=\bar A_t-\eta M_A$, and $X_{t+1}=B_{t+1}A_{t+1}$.

Lower row:

1. **Gauge symmetry and balancing:** show two different factor pairs making the same matrix: $(B,A)\sim(BQ,Q^{-1}A)$ and $(BQ)(Q^{-1}A)=BA$.
2. **State transport before projection:** show $M_{t-1}$ becoming $\widetilde M_{t-1}$, then $U$, then $\Omega$. Include $\widetilde M_B=M_{B,t-1}Q$ and $\widetilde M_A=Q^{-1}M_{A,t-1}$.
3. **Horizontal update:** show $U$ splitting into the projected update $M$ and factor-only motion. Include $\bar B^\top M_B-M_A\bar A^\top=2\varepsilon\Omega$ and a purple $\Omega$ tile. This term is zero when $\varepsilon=0$.

Use the reference for appearance only. Check every label and equation against the paper before replacing the image.
