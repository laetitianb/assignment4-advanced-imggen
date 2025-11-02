# Assignment 4

## 1 Practice: Model Deployment
Use CIFAR-10 (32×32, RGB) for both models and add endpoints to the Module-6 API.
- Diffusion: UNet predicting ε; sampler exposes `img_size=32, channels=3`.
- EBM: energy network; Langevin on **inputs** (x.requires_grad_(True)).
- Expose `/generate/diffusion` and `/generate/ebm` and commit to GitHub.

## 2 Theory: Building Blocks of a Diffusion Model

**Question 1**  
Sinusoidal time embedding for dimension \(i\) (0-indexed), embedding dim \(d\), max period \(P\):  
\[
\text{emb}_i(t)=
\begin{cases}
\sin\!\big(t / P^{\frac{2k}{d}}\big), & i=2k\\[4pt]
\cos\!\big(t / P^{\frac{2k}{d}}\big), & i=2k+1
\end{cases}
\]

**Question 2**  
\(d=8,\ t=1,\ P=10000.\) Let \(w_k=P^{-2k/d}=10^{-4k/8}=\{1,0.3162278,0.1,0.0316228\}\).  
Embedding ≈ \([0.841471,\ 0.540302,\ 0.310268,\ 0.950650,\ 0.099833,\ 0.995004,\ 0.031622,\ 0.999500]\).

**Question 3**  
Both use the same sinusoidal recipe; **Transformers** encode token *positions* for sequence order, **diffusion** encodes *noise level/time step* \(t\) to condition denoising strength.

**Question 4**  
64×64 with 3 stride-2 downsamples → **8×8**.

**Question 5**  
UNet outputs \(\hat\epsilon(x_t,t)\) (same shape as \(x_t\)); loss is MSE:  
\(\ \mathcal L=\|\epsilon-\hat\epsilon\|^2\), with \(x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon\).

## 3 Theory: Building Blocks of an Energy Model

**Question 6: Basic Gradient Calculations**
```python
import torch
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x
y.backward()
print(x.grad)  # tensor([7.])


