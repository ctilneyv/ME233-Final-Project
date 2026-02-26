# ME233 Final Project
## Neural-Network Augmentation of Nonlinear Parametric Reduced-Order Modeling of 2D Advection-Diffusion

---

## Overview

This project implements and benchmarks a hierarchy of reduced-order modeling (ROM) strategies for a **2D steady-state advection-diffusion equation** with **temperature-dependent diffusivity**. The governing PDE introduces a physically motivated nonlinearity that breaks the affine approximation dependence assumed by classical (linear) ROMs, such as Galerkin and Petrov-Galerkin Projection Methods. This makes it an ideal testbed for nonlinear model reduction techniques.

While the Kolmogorov barrier motivates the development of nonlinear ROM strategies for transport-dominated problems, this project uses a nonlinear diffusion testcase to isolate and compare two fundamentally different approaches to handling nonlinearity in reduced-order models: physics-informed hyper-reduction (DEIM) and data-driven latent-space correction (Artificial Neural Network (ANN)-augmented PMOR).

---

## Problem Statement

### Governing PDE

The steady-state (SS) advection-diffusion equation is solved on the unit square domain $\Omega = [0,1]^2$:

$$U \frac{\partial T}{\partial x} = \nabla \cdot \left( \kappa(T) \nabla T \right)$$

where the diffusivity is **temperature-dependent**:

$$\kappa(T) = \kappa_0 \left(1 + \alpha (T - T_{\text{ref}}) \right)$$

This nonlinearity is physically motivated. Thermal conductivity in many materials (metals, gases) varies with local temperature. Hotter regions diffuse heat more aggressively than cooler ones, coupling the thermal and flow fields nonlinearly.

### Boundary Conditions

| Boundary | Condition |
|---|---|
| $x = 0$ (left wall) | Dirichlet: parametric hotspot profile $T_D(y; \bar{y})$ |
| $x = 1$ (right wall) | Neumann: $\partial T / \partial x = 0$ |
| $y = 0, 1$ (top/bottom) | Neumann: $\partial T / \partial y = 0$ |

The left wall Dirichlet profile is a smooth sinusoidal hotspot centered at $\bar{y}$:

| Region | $T_D(y;\, \bar{y})$ |
|---|---|
| $y < 1/3$ | $300$ |
| $1/3 \leq y \leq 2/3$ | $300 + 325\left(\sin\left(3\pi\|y - \bar{y}\|\right) + 1\right)$ |
| $y > 2/3$ | $300$ |

producing temperatures in the range $[300, 950]$ K.

<img width="1120" height="674" alt="image" src="https://github.com/user-attachments/assets/8e692572-9ddf-4ec7-b948-105cde1ac028" />

### Parameter Domain

The problem is parametrized by $\boldsymbol{\mu} = [U\, \kappa_0\, \bar{y}]$:

| Parameter | Description | Range |
|---|---|---|
| $U$ | Convective velocity | $[0.1\, 0.6]$ |
| $\kappa_0$ | Reference diffusivity | $[5 \times 10^{-3}\, 0.025]$ |
| $\bar{y}$ | Hotspot center location | $[0.4\, 0.6]$ |

The Péclet number $Pe = UL/\kappa_0$ ranges up to 120, spanning diffusion-dominated to moderately convection-dominated regimes.

---

## Discretization

The PDE is discretized on a **75 × 75 finite-difference grid** using:
- **Second-order central differences** for the diffusion term $\nabla \cdot (\kappa \nabla T)$, with arithmetic face averaging of $\kappa$ between adjacent nodes
- **First-order upwind differencing** for the convection term $U \partial T / \partial x$

This yields a linear system for the $73 \times 73 = 5{,}329$ interior degrees of freedom:

$$A(\boldsymbol{\mu})\ \mathbf{w} = \mathbf{b}(\boldsymbol{\mu})$$

where $A$ depends nonlinearly on the solution $\mathbf{w}$ through $\kappa(T)$.

---

## High-Dimensional Model (HDM)

Because $\kappa$ depends on $T$, the system cannot be solved directly. The HDM is solved via **Picard (successive substitution) iteration**:

1. Warm-start initialization by computing the linear solution at $\kappa_0$
2. Compute the spatially-averaged effective diffusivity $\bar{\kappa}^{(n)} = \langle \kappa_0(1 + \alpha(T^{(n)} - T_{\text{ref}}))\rangle$
3. Reassemble and solve the linear system at $\bar{\kappa}^{(n)}$
4. Apply damped update $\mathbf{w} \leftarrow (1-\omega)\mathbf{w}^{(n)} + \omega\mathbf{w}^{(n+1)}$ with adaptive $\omega \in [0.05\, 0.5]$
5. Repeat until $\|\mathbf{w}^{(n+1)} - \mathbf{w}^{(n)}\| / \|\mathbf{w}^{(n)}\| < \varepsilon$

A snapshot database of **1,000 HDM solutions** is generated over a full-grid parameter sweep, all converging without failure.

---

## Reduced-Order Modeling

The solution is approximated by the subspace ansatz:

$$\tilde{\mathbf{w}} = \mathbf{w}_{\text{ref}} + \mathbf{V}\mathbf{q}, \quad \mathbf{V} \in \mathbb{R}^{N \times k}\; \mathbf{q} \in \mathbb{R}^k\; k \ll N$$

where $\mathbf{V}$ is a **Reduced-Order Basis (ROB)** built from solution snapshots via the **method of snapshots (POD)**. Applying an orthogonal Galerkin projection $\Pi_{\mathbf{V},\mathbf{V}} = \mathbf{V}\mathbf{V}^T$ – that is, constraining the residual to be orthogonal to $\text{range}(\mathbf{V})$ – yields the reduced system:

$$A_r \mathbf{q} = \mathbf{b}_r, \quad A_r = \mathbf{V}^T A \mathbf{V} \in \mathbb{R}^{k \times k}\quad \mathbf{b}_r = \mathbf{V}^T \mathbf{b} \in \mathbb{R}^k$$

The POD energy spectrum of the nonlinear snapshot database decays quickly — 99.9% of energy is captured in a handful of modes — indicating a reasonably low-dimensional solution subspace $\mathcal{M}$ despite the nonlinearity.

### Error Metric

All ROM errors are reported as the relative $\ell^2$ error:

$$\varepsilon(\boldsymbol{\mu}) = \frac{\left\|\mathbf{w}_{\text{HDM}} - \mathbf{w}_{\text{ROM}}\right\|_2}{\left\|\mathbf{w}_{\text{HDM}}\right\|_2}$$

---

## Part 2: Linear PROM — Kolmogorov Barrier Study

### Motivation

Before introducing nonlinear correction strategies, it is important to establish the fundamental limitation of the linear ROM on this problem. The **Kolmogorov $n$-width** $d_n(\mathcal{M})$ measures the best possible approximation error achievable by *any* $n$-dimensional linear subspace over the solution manifold $\mathcal{M}$:

$$d_n(\mathcal{M}) = \inf_{\mathbf{V} \in \mathbb{R}^{N \times n}} \sup_{\mathbf{w} \in \mathcal{M}} \inf_{\mathbf{q} \in \mathbb{R}^n} \left\|\mathbf{w} - \mathbf{V}\mathbf{q}\right\|_2$$

If $d_n(\mathcal{M})$ decays slowly with $n$, no linear ROM — regardless of how many snapshots are used — can achieve low error. This is the **Kolmogorov barrier**.

### Snapshot Budget Sweep

To empirically measure this barrier, Part 2 sweeps the snapshot budget $k$ from 1 upward. For each $k$:

1. Draw $k$ parameter points via **Latin Hypercube Sampling (LHS)** of $\mathcal{D}$
2. Snap each to the nearest point in the precomputed HDM database
3. Build a $k$-mode POD basis $\mathbf{V}_k$ from those $k$ snapshots
4. Check the condition number $\kappa(A_r) = \text{rcond}(\mathbf{V}_k^T A \mathbf{V}_k)^{-1}$ of the reduced operator
5. If $\text{rcond} < 10^{-12}$, the basis has become numerically singular — stop
6. Otherwise evaluate on all remaining test points and record $E_{\text{max}}$, $E_{\text{avg}}$

The offline cost at each $k$ is reported honestly as:

$$t_{\text{offline}}(k) = k \cdot t_{\text{HDM}} + t_{\text{POD}}$$

where $t_{\text{HDM}}$ is the average cost of a single HDM solve and $t_{\text{POD}}$ is the basis construction time.

### Result: The Barrier

The sweep reveals that error saturates quickly and decreases very slowly, even as $k$ grows toward the singularity threshold. The reduced operator becomes nearly rank deficient, setting a near upper limit on the linear ROM's snapshot budget. 

This is empirical evidence of the Kolmogorov barrier: the solution manifold $\mathcal{M}$ induced by the temperature-dependent diffusivity $\kappa(T)$ cannot be well-approximated by any linear subspace of tractable dimension. No amount of additional snapshots resolves this — the barrier is fundamental, not a sampling artifact.

This result directly motivates the nonlinear correction strategies in Parts 3 and 4.

---

## Hyper-Reduction: Discrete Empirical Interpolation Method (DEIM)

The affine ROM assembles the system matrix at a fixed reference diffusivity and ignores how $\kappa$ varies with temperature across the domain. DEIM corrects this by learning where to sample the nonlinear field so it can be reconstructed globally from just a few spatial evaluations.

### Offline: Nonlinear Snapshot Collection

For each HDM snapshot, the diffusivity is evaluated at every interior node to build a nonlinear snapshot matrix. POD on this matrix yields a low-dimensional DEIM basis. Since $\kappa(T)$ is affine in $T$, only 2 basis modes are needed to capture 99.99% of the variance across all 1,000 parameter points.

### Offline: Greedy Index Selection

Although a greedy algorithm would be useful, a basis size parameter identifies the 2 most informative spatial locations (or anything on the order of 1-10) and builds a mask matrix $P$. The DEIM projection operator is then precomputed:

$$\Pi_f = \mathbf{V}_f (P^T \mathbf{V}_f)^{-1}$$

### Online: Query Evaluation

At a new parameter point, the diffusivity field is reconstructed from evaluations at only 2 spatial indices:

$$\kappa(\mathbf{w}) \approx \Pi_f \cdot \kappa(\mathbf{w}_{\mathcal{I}})$$

The spatial mean of this reconstructed field gives an informed effective diffusivity, which replaces the naive $\kappa_0$ used by the affine ROM when assembling the reduced system.

---

## ANN-Augmented PMOR

### Motivation

DEIM corrects the nonlinearity by sampling the diffusivity field at sparse spatial locations. The ANN-augmented approach takes a fundamentally different path: rather than correcting the physics operator, it learns a **closure map in reduced coordinates** that captures what the primary linear subspace cannot represent.

The starting point is the augmented solution manifold from Barnett, Farhat & Maday (2023):

$$\tilde{\mathbf{w}} = \mathbf{w}_{\text{ref}} + \mathbf{V}\mathbf{q} + \bar{\mathbf{V}}\mathcal{N}(\mathbf{q})$$

where $\mathbf{V} \in \mathbb{R}^{N \times k}$ is the primary basis (modes 1–$k$), $\bar{\mathbf{V}} \in \mathbb{R}^{N \times \bar{k}}$ is a secondary basis (modes $k+1$ through $k+\bar{k}$), and $\mathcal{N}: \mathbb{R}^k \to \mathbb{R}^{\bar{k}}$ is a neural network that predicts the secondary coordinates from the primary ones. The affine ROM uses only the first two terms and stops. The ANN adds the correction $\bar{\mathbf{V}}\bar{\mathbf{q}}_{\text{pred}}$ on top.

### Interpretation

This method can be interpreted simultaneously as:

- A **latent-space constitutive model** — $\mathcal{N}(\mathbf{q})$ plays the role of a closure law, relating coarse (primary) and fine (secondary) reduced coordinates, analogous to a turbulence closure or subgrid stress model
- A **physics-embedded nonlinear correction** — the architecture is constrained to reflect the known quadratic structure of the governing nonlinearity (see below)
- A **constrained automated model discovery step** — rather than discovering the full solution map $\boldsymbol{\mu} \to \mathbf{w}$, the network only needs to learn the residual between the linear ROM and the true solution, projected onto $\bar{\mathbf{V}}$

### Neural Closure Structure

Expanding the governing PDE, the true solution satisfies:

$$\nabla \cdot \left(\kappa(T) \nabla T\right) = U \frac{\partial T}{\partial x}, \quad \kappa(T) = \kappa_0\left(1 + \alpha(T - T_{\text{ref}})\right)$$

Expanding the diffusion term:

$$\kappa_0 \nabla^2 T + \underbrace{\kappa_0 \alpha (T - T_{\text{ref}}) \nabla^2 T + \kappa_0 \alpha \|\nabla T\|^2}_{\text{nonlinear correction}} = U \frac{\partial T}{\partial x}$$

The affine ROM discards the nonlinear correction entirely. The closure $\bar{\mathbf{q}} \in \mathbb{R}^{\bar{k}}$ is precisely the projection of this correction onto $\bar{\mathbf{V}}$. Substituting the ROM ansatz $T \approx T_{\text{ref}} + \mathbf{V}\mathbf{q}$:

$$\underbrace{\kappa_0 \alpha (\mathbf{V}\mathbf{q}) \nabla^2 (\mathbf{V}\mathbf{q})}_{\text{linear in } \mathbf{q}} + \underbrace{\kappa_0 \alpha \nabla(\mathbf{V}\mathbf{q}) \cdot \nabla(\mathbf{V}\mathbf{q})}_{\text{quadratic in } \mathbf{q}}$$

The first term is linear in $\mathbf{q} \in \mathbb{R}^k$ and is already captured by a standard hidden layer. The second term scales like $\mathbf{q} \odot \mathbf{q}$ — the physics directly predicts a quadratic component in the closure $\bar{\mathbf{q}} \in \mathbb{R}^{\bar{k}}$. The network architecture is therefore constrained to reflect this:

$$\bar{\mathbf{q}} \approx \underbrace{W_2 \tanh(W_1 \mathbf{q} + \mathbf{b}_1) + \mathbf{b}_2}_{\text{standard hidden layer}} + \underbrace{W_3 (\mathbf{q} \odot \mathbf{q})}_{\text{physics skip: encodes } \|\nabla T\|^2}$$

$W_3$ is not regularization — it encodes the specific quadratic structure of $\|\nabla T\|^2$ that the affine ROM neglects. This is analogous to the **CANN (Constrained Artificial Neural Network)** philosophy: identify the terms the physics predicts, build them explicitly into the architecture, and let the data tune the coefficients rather than discover the structure from scratch.

### Inputs, Outputs, and Network Architecture

**Input**: $\mathbf{q} \in \mathbb{R}^k$ — the primary reduced coordinates, computed as $\mathbf{q} = \mathbf{V}^T(\mathbf{w} - \mathbf{w}_{\text{ref}})$. These encode where in the $k$-dimensional primary subspace the solution lives for a given $\boldsymbol{\mu}$.

**Output**: $\bar{\mathbf{q}} \in \mathbb{R}^{\bar{k}}$ — the secondary reduced coordinates, $\bar{\mathbf{q}} = \bar{\mathbf{V}}^T(\mathbf{w} - \mathbf{w}_{\text{ref}})$. These encode the closure error — the component of the solution that the primary basis misses. At test time the network predicts $\bar{\mathbf{q}}$ directly from $\mathbf{q}$, with no additional HDM solve required.

Substituting the full augmented ansatz $\tilde{\mathbf{w}} = \mathbf{w}_{\text{ref}} + \mathbf{V}\mathbf{q} + \bar{\mathbf{V}}\bar{\mathbf{q}}$ into the governing PDE and projecting onto $\text{range}(\mathbf{V})$ yields:

$$\underbrace{\kappa_0 \mathbf{V}^T \nabla^2 (\mathbf{V}\mathbf{q})}_{\text{affine ROM term}} + \underbrace{\kappa_0 \alpha\, \mathbf{V}^T \left[ (\mathbf{V}\mathbf{q} + \bar{\mathbf{V}}\bar{\mathbf{q}}) \nabla^2 (\mathbf{V}\mathbf{q} + \bar{\mathbf{V}}\bar{\mathbf{q}}) + \nabla(\mathbf{V}\mathbf{q} + \bar{\mathbf{V}}\bar{\mathbf{q}}) \cdot \nabla(\mathbf{V}\mathbf{q} + \bar{\mathbf{V}}\bar{\mathbf{q}}) \right]}_{\text{nonlinear correction} \approx \mathbf{V}^T \bar{\mathbf{V}} \mathcal{N}(\mathbf{q}) = \mathcal{N}(\mathbf{q})\text{ since } \mathbf{V}^T\bar{\mathbf{V}}=0} = \mathbf{V}^T U \frac{\partial}{\partial x}(\mathbf{V}\mathbf{q} + \bar{\mathbf{V}}\bar{\mathbf{q}})$$

where $\bar{\mathbf{q}} = \mathcal{N}(\mathbf{q}) = W_2 \tanh(W_1\mathbf{q} + \mathbf{b}_1) + \mathbf{b}_2 + W_3(\mathbf{q} \odot \mathbf{q})$. The orthogonality $\mathbf{V}^T\bar{\mathbf{V}} = 0$ ensures the primary and secondary subspaces decouple cleanly, so the network correction enters the projected residual without polluting the primary Galerkin equations.

| Layer | Operation | Output size | Parameters |
|---|---|---|---|
| Input | $\mathbf{q}$ | $k = 6$ | — |
| Hidden | $\tanh(W_1 \mathbf{q} + \mathbf{b}_1)$ | $h = 32$ | $W_1 \in \mathbb{R}^{32 \times 6}$, $\mathbf{b}_1 \in \mathbb{R}^{32}$ |
| Output | $W_2 \mathbf{h} + \mathbf{b}_2$ | $\bar{k} = 14$ | $W_2 \in \mathbb{R}^{14 \times 32}$, $\mathbf{b}_2 \in \mathbb{R}^{14}$ |
| Physics skip | $W_3 (\mathbf{q} \odot \mathbf{q})$ | $\bar{k} = 14$ | $W_3 \in \mathbb{R}^{14 \times 6}$ |
| **Total** | | | **770 parameters** |

Training minimizes the MSE loss with L2 regularization:

$$\mathcal{L} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \left\|\bar{\mathbf{q}}_i - \mathcal{N}(\mathbf{q}_i)\right\|_2^2 + \lambda \left(\|W_1\|_F^2 + \|W_2\|_F^2 + \|W_3\|_F^2\right)$$

with Adam optimizer, learning rate $10^{-3}$, batch size 128, and $\lambda = 10^{-4}$.

## Repository Structure

```
.
├── main.m                  # Main script (Parts 1–4)
├── functions/              # Core solver and ROM functions
│   ├── computeNonLinearHDM.m
│   ├── assembleLinearSystem.m
│   ├── assembleNonLinearSystem.m
│   ├── buildBasisPOD.m
│   ├── assembleROM.m
│   ├── computeROM.m
│   ├── computeErrors.m
│   ├── reshapeSolution.m
│   ├── computeNonLinearKappa.m
│   └── buildMaskDEIM.m
├── visualization/
│   ├── plotSolution.mlx
│   ├── plotErrorComparison.mlx
│   └── plotTimeComparison.mlx
├── *.mat                   # Saved HDM snapshots and metadata
└── README.md
```

---

## References

- Farhat, C. *AA 216 / CME 345: Projection-Based Model Order Reduction*, Stanford University, 2024.
- Barnett, J., Farhat, C., and Maday, Y. "Neural-network-augmented projection-based model order reduction for mitigating the Kolmogorov barrier to reducibility." *Journal of Computational Physics*, 492:112420, 2023.
- Chaturantabut, S. and Sorensen, D.C. "Nonlinear model reduction via discrete empirical interpolation." *SIAM Journal on Scientific Computing*, 32(5):2737–2764, 2010.
