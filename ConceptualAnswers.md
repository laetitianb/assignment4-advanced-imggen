# Assignment 4 — Conceptual / Calculation Answers

## Diffusion

**Q1. Forward process q(x_t|x_0): mean/var**
\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal N(0,I)
\]
Mean \(=\sqrt{\bar\alpha_t}x_0\). Var \(=(1-\bar\alpha_t)I\).

**Q2. \(\bar\alpha_t\), \(\alpha_t\), \(\beta_t\)**
\[
\alpha_t=1-\beta_t,\quad \bar\alpha_t=\prod_{s=1}^t \alpha_s
\]

**Q3. Reverse mean with \(\epsilon_\theta\)**
\[
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)
\]

**Q4. Posterior variance**
\[
\tilde\beta_t=\frac{1-\bar\alpha_t}{1-\bar\alpha_{t-1}}\beta_t
\]

**Q5. Loss**
\[
\mathcal L=\mathbb E[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\|^2]
\]

**Q6. Time embedding (sin/cos)**
\[
\text{emb}[2k]=\sin\!\big(t/P^{2k/d}\big),\quad \text{emb}[2k+1]=\cos\!\big(t/P^{2k/d}\big)
\]

**Q7. Example (d=8, t=1, P=10000)**
\(w_k=10^{-4k/8}=\{1,10^{-0.5},10^{-1},10^{-1.5}\}\). Vector:
\([\sin w_0,\cos w_0,\sin w_1,\cos w_1,\sin w_2,\cos w_2,\sin w_3,\cos w_3]\).

**Q8. Why time emb helps**
It conditions the UNet on noise level \(t\).

**Q9. SNR**
\(\text{SNR}_t=\bar\alpha_t/(1-\bar\alpha_t)\) decreases with \(t\).

**Q10. Downsample 64→8**
Three stride-2 downs: 64→32→16→8.

**Q11. UNet I/O + loss**
Input \((x_t,t)\), output \(\hat\epsilon\); loss is MSE to true \(\epsilon\).

**Q12. Class guidance (cfg)**
\(\hat\epsilon=\epsilon_\theta(x_t,\emptyset)+w(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\emptyset))\).

**Q13. \(\beta_t\) schedule**
Linear works; cosine often better.

**Q14. Sampling cost**
\(\mathcal O(T)\) network evals.

**Q15. Deterministic vs stochastic**
DDIM sets variance→0 (faster, less diversity).

## EBMs

**Q16. Objective**
Lower energy for data than non-data (contrastive/score objectives).

**Q17. Langevin**
\(x \leftarrow x-\eta\nabla_x E_\theta(x)+\sigma\xi,\; \xi\sim\mathcal N(0,I)\).

**Q18. Grads needed**
Sampling: inputs only; Training: params.

**Q19. Noise role**
Prevents mode collapse; improves mixing.

**Q20. Accumulation**
Zero `x.grad` each step.

**Q21. Grad check**
\(y=x^2+3x\Rightarrow \nabla_x y=2x+3\). At \(x=2\), 7.

**Q22. With weights**
\(y=w^\top x\Rightarrow \nabla_w y=x,\; \nabla_x y=w\).

**Q23. Detach**
`detach()` stops grads; remove it to keep flow.

**Q24. Clamp**
Clip \(x\) to bounds after updates.

**Q25. Step/noise tradeoff**
Tune \(\eta,\sigma\) for stability vs exploration.

## DDPM vs EBM

**Q26. What’s updated**
DDPM: latent \(x_t\) via reverse mean. EBM: input \(x\) via \(-\nabla_x E\).

**Q27. Signal**
DDPM: MSE on noise. EBM: energy comparison.

**Q28. Cost**
Both iterative; acceleration needed for speed.

**Q29. Conditioning**
Both support; implemented differently.

**Q30. Fail modes**
DDPM: blur if undertrained. EBM: mode collapse/poor mixing.
