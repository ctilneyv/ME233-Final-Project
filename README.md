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

The POD energy spectrum of the nonlinear snapshot database decays quickly — 99.88% of energy is captured in $k = 6$ modes — indicating a low-dimensional solution manifold $\mathcal{M}$ despite the nonlinearity.

---

## Part 2: Snapshot Budget Study (Updated)

Part 2 now performs a systematic **snapshot budget loop** rather than a single ROM construction.

For varying training snapshot counts

$$
n_s \in \{5, 10, 20, 40, 80\}
$$

the workflow is repeated over multiple independent LHS trials:

1. Sample training parameters
2. Build POD basis
3. Construct ROM (affine, DEIM, or ANN-augmented)
4. Evaluate over a validation set
5. Report mean ± standard deviation of relative $\ell^2$ error

This produces statistical convergence trends of ROM accuracy with respect to snapshot count.

---

### Error Metric

All ROM errors are reported as the relative $\ell^2$ error:

$$\varepsilon(\boldsymbol{\mu}) = \frac{\left\|\mathbf{w}_{\text{HDM}} - \mathbf{w}_{\text{ROM}}\right\|_2}{\left\|\mathbf{w}_{\text{HDM}}\right\|_2}$$

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

---

## ANN-Augmented Projection-Based Model Order Reduction (Part 4)

The nonlinearity in the governing PDE arises from the temperature-dependent diffusivity

$$
\kappa(T) = \kappa_0\left(1 + \alpha (T - T_{\text{ref}})\right),
$$

which induces a nonlinear diffusion operator

$$
\nabla \cdot \big( \kappa(T) \nabla T \big).
$$

After discretization and Galerkin projection onto the POD basis $\mathbf{V}$, the reduced solution is

$$
\tilde{\mathbf{w}} = \mathbf{w}_{\text{ref}} + \mathbf{V}\mathbf{q},
$$

and the reduced system contains a nonlinear term inherited from $\kappa(T)$.  
Because $\kappa(T)$ is affine in $T$, substitution of the reduced ansatz yields a **quadratic dependence in the reduced coordinates**:

$$
\kappa(\tilde{\mathbf{w}})
=
\kappa_0
+
\alpha \kappa_0 \mathbf{V}\mathbf{q}.
$$

Thus, although the diffusivity is linear in temperature, the projected operator becomes nonlinear in $\mathbf{q}$ due to the coupling between $\kappa(\mathbf{w})$ and $\nabla \mathbf{w}$ inside the divergence operator. This induces curvature in the reduced manifold $\mathcal{M}$ that is not captured by an affine ROM.

---

### Latent-Space Representation of the Nonlinear Term

To isolate this nonlinear contribution, snapshots of the nonlinear component are projected onto a secondary basis $\bar{\mathbf{V}}$:

$$
\mathbf{g}(\mathbf{w}) = \nabla \cdot \big(\kappa(\mathbf{w}) \nabla \mathbf{w}\big)
\quad \longrightarrow \quad
\mathbf{g} \approx \bar{\mathbf{V}} \bar{\mathbf{q}}.
$$

Here:

- $\mathbf{V}$ spans the **solution manifold**
- $\bar{\mathbf{V}}$ spans the **nonlinear operator manifold**
- $\mathbf{q}$ are solution coordinates
- $\bar{\mathbf{q}}$ are nonlinear operator coordinates

In classical hyper-reduction (e.g., DEIM), $\bar{\mathbf{q}}$ is recovered through sparse spatial sampling.

In Part 4, instead of reconstructing the nonlinear field, we directly learn the mapping in latent space.

---

### Neural Network Closure in Reduced Coordinates

The ANN learns the parametric mapping

$$
\boldsymbol{\mu}
\;\longrightarrow\;
\widehat{\bar{\mathbf{q}}}
$$

where:

- Input: parameter vector $\boldsymbol{\mu} = [U,\kappa_0,\bar{y}]$
- Output: reduced nonlinear coordinates $\widehat{\bar{\mathbf{q}}}$

These coordinates represent the projected nonlinear diffusion contribution in the reduced operator basis $\bar{\mathbf{V}}$.

The reduced system therefore becomes

$$
\mathbf{A}_r \mathbf{q}
+
\bar{\mathbf{V}} \widehat{\bar{\mathbf{q}}}
=
\mathbf{b}_r,
$$

where the ANN provides a closure model for the nonlinear term directly in reduced space.

---

### Connection to the Quadratic Structure

Because the original nonlinearity is linear in $T$ but appears inside a divergence operator, the reduced dynamics exhibit effective quadratic dependence in $\mathbf{q}$.

The ANN does not replace the PDE or learn the solution directly.  
It approximates the nonlinear reduced operator coordinates $\bar{\mathbf{q}}$, which encode this quadratic interaction in latent space.

Thus:

- $\mathbf{V}$ captures the dominant solution manifold.
- $\bar{\mathbf{V}}$ captures the dominant nonlinear operator manifold.
- The ANN learns how parameters excite nonlinear modes in $\bar{\mathbf{V}}$.

This preserves:

- The Galerkin projection structure,
- The governing PDE operator form,
- The physics of diffusion–convection balance,

while replacing the expensive nonlinear evaluation with a learned reduced closure.

---

### Interpretation

This method can be interpreted as:

- A **latent-space constitutive model**,
- A physics-embedded nonlinear closure,
- A constrained automated model discovery step in reduced coordinates.

The governing equations remain fixed.  
The neural network only models how the nonlinear operator projects onto the reduced manifold.


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
