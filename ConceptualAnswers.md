# Assignment 4 — Conceptual / Calculation Answers (Clear + Explicit)
## Diffusion
1) x_t=√(ᾱ_t)x_0+√(1−ᾱ_t)ε; mean=√(ᾱ_t)x_0; var=(1−ᾱ_t)I
2) α_t=1−β_t; ᾱ_t=∏_{s=1}^t α_s
3) μ_θ(x_t,t)=1/√(α_t)[x_t−(β_t/√(1−ᾱ_t))ε_θ(x_t,t)]
4) \tildeβ_t=((1−ᾱ_t)/(1−ᾱ_{t−1}))β_t
5) L=E[||ε−ε_θ(√(ᾱ_t)x_0+√(1−ᾱ_t)ε,t)||^2]
6) emb[2k]=sin(t/P^{2k/d}); emb[2k+1]=cos(t/P^{2k/d})
7) d=8,t=1,P=10000 → [0.841471,0.540302,0.310268,0.950650,0.099833,0.995004,0.031622,0.999500]
8) Time emb tells UNet noise level per step.
9) SNR_t=ᾱ_t/(1−ᾱ_t) ↓ with t.
10) 64→32→16→8 after 3 stride-2 downs.
11) In: (x_t,t). Out: ε̂. Loss: MSE(ε,ε̂).
12) CFG: ε̂=ε_θ(x_t,∅)+w(ε_θ(x_t,c)−ε_θ(x_t,∅))
13) β schedule: linear OK; cosine often better.
14) O(T) evals; fewer steps=faster/lower quality.
15) DDIM (var→0) deterministic; DDPM stochastic.

## EBMs
16) Learn E so data low energy, non-data high (contrastive/score-like).
17) x←x−η∇_xE_θ(x)+σξ; set x.requires_grad_(True)
18) Sampling: grads on inputs; Training: grads on params.
19) Noise improves mixing/avoids traps.
20) Zero x.grad each step (no accumulation).
21) y=x^2+3x ⇒ dy/dx=2x+3; at x=2 → 7.
22) y=w^T x ⇒ ∇_w y=x, ∇_x y=w.
23) detach() blocks grads; remove to keep flow.
24) Clamp x (e.g., [-3,3] or [0,1]).
25) Tune η,σ for stability vs exploration.

## DDPM vs EBM
26) DDPM updates x_t via reverse mean; EBM updates x via −∇_xE.
27) DDPM supervised MSE; EBM energy shaping (harder).
28) Both iterative; DDPM many steps unless accelerated; EBM mixing depends on steps/η/σ.
29) Both conditionable (implemented differently).
30) DDPM blur if undertrained/few steps; EBM mode collapse/poor mixing.
