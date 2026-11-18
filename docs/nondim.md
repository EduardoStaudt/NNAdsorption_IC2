# Nondimensionalization of the v23b model

This note derives the dimensionless equations used by the PINN residuals
(`src/nnads_ic2/physics/residuals.py`) from Eqs. (1)–(6) of the solver v23b preview
(`docs/modelo_solver_v23b_preview.pdf`). Code: `src/nnads_ic2/physics/scales.py`.

Why bother: the raw variables differ by many orders of magnitude (c ~ 10³ mol/m³,
q ~ 1 mol/kg, T ~ 300 K, t ~ 10–3600 s, z ~ 1 m). A network learns best when inputs and
outputs are O(1), and the loss terms can only be balanced when every residual is O(1).

## 1. Scales

| Variable | Dimensionless | Scale | Meaning of the scale |
|---|---|---|---|
| z | z̃ = z/L | L | bed length |
| t | t̃ = t/t_ref | t_ref = max_i t_st,i | stoichiometric time of the slowest front, Eq. (7) |
| c_i | c̃_i = c_i/C_F | C_F = P/(R T_in) | total gas concentration of the feed |
| q̄_i | q̃_i = q̄_i/q_ref,i | q_ref,i = max(q*_i,F, q*_i,0, floor) | largest equilibrium loading the component can have |
| T | T̃ = (T − T_in)/ΔT_ref | ΔT_ref = max(1 K, r ΔT_feed), ΔT_feed = Σ_i y_i,F (−ΔH_i) q_ref,i/C_p,s | size of the temperature change (see the note below) |
| v | ṽ = v/v_s | v_s | feed superficial velocity |

with the stoichiometric time (Eq. 7)

$$t_{st,i} = \frac{L}{v_s}\left(\varepsilon_b + \rho_b \frac{q^*_{i,F}}{c_{i,F}}\right),\qquad c_{i,F} = y_{i,F}\,C_F .$$

Notes:

- Temperature scale. ΔT_feed is the heat the solid would release taking up the feed loading
  with no losses, over its heat capacity. It ignores the heat carried away by the gas, the
  cooling by desorption of the initial gas and the wall, so it can overestimate the real change
  several times (Section 7: 33.6 K against a measured 4.85 K). When reference data exist,
  r = (measured max|T − T_in|) / ΔT_feed:
  - one case (Phase 0): ΔT_ref is the measured change itself;
  - several cases (Phase 1): r is the median over the training cases, so ΔT_ref stays a
    function of the case parameters alone and validation/holdout cases (no data at prediction
    time) follow the same rule;
  - no data: r = 1 (ΔT_feed), with a warning in the log.
  The value of r is saved in the run's `config.yaml` (`scales.dT_ratio`).

- q*_i,F is the loading in equilibrium with the feed and q*_i,0 the loading in equilibrium with
  the gas that fills the bed at t = 0. Taking the larger of the two gives the right scale to
  every component without a fixed "carrier" index: the initial gas gets its initial loading,
  the fed components their feed loading. A component absent from both gets the floor.
- The isotherm uses partial pressures in **atm** (B in 1/atm), exactly like IC1
  (`p_i = c_i R T / 101325`).
- (−ΔH_i) is stored as a positive number (`dH`), as in IC1.

## 2. Substitution rule

Every derivative picks up the ratio of scales, e.g.

$$\frac{\partial c_i}{\partial t} = \frac{C_F}{t_{ref}}\frac{\partial \tilde c_i}{\partial \tilde t},\qquad
\frac{\partial^2 c_i}{\partial z^2} = \frac{C_F}{L^2}\frac{\partial^2 \tilde c_i}{\partial \tilde z^2},\qquad
\frac{\partial T}{\partial z} = \frac{\Delta T_{ref}}{L}\frac{\partial \tilde T}{\partial \tilde z}.$$

## 3. Equations

### 3.1 Gas-phase mass balance, Eq. (1)

$$\varepsilon_b \frac{\partial c_i}{\partial t} + \frac{\partial (v c_i)}{\partial z} = \varepsilon_b D_L \frac{\partial^2 c_i}{\partial z^2} - \rho_b \frac{\partial \bar q_i}{\partial t}$$

Substitute the scales and multiply by L/(v_s C_F) (the convective scale):

$$\alpha\,\frac{\partial \tilde c_i}{\partial \tilde t} + \frac{\partial (\tilde v \tilde c_i)}{\partial \tilde z}
- \frac{1}{Pe}\frac{\partial^2 \tilde c_i}{\partial \tilde z^2} + \beta_i \frac{\partial \tilde q_i}{\partial \tilde t} = 0$$

| Group | Formula | Meaning |
|---|---|---|
| α | ε_b L/(v_s t_ref) | gas residence time / t_ref (≤ 1 because t_ref ≥ ε_b L/v_s) |
| Pe | v_s L/(ε_b D_L) | Péclet: convection / axial dispersion |
| K_i | ρ_b q_ref,i/(ε_b C_F) | capacity ratio: adsorbed holdup / gas holdup |
| β_i | ρ_b q_ref,i L/(v_s C_F t_ref) = α K_i | uptake term |

The product rule gives ∂(ṽ c̃_i)/∂z̃ = ṽ ∂c̃_i/∂z̃ + c̃_i ∂ṽ/∂z̃.

### 3.2 LDF, Eq. (3)

$$\frac{\partial \bar q_i}{\partial t} = k_{L,i}\,(q^*_i - \bar q_i)
\quad\Longrightarrow\quad
\frac{\partial \tilde q_i}{\partial \tilde t} = Da_i\,(\tilde q^*_i - \tilde q_i),\qquad Da_i = k_{L,i}\,t_{ref}$$

with q̃*_i = q*_i(C_F c̃, T_in + ΔT_ref T̃)/q_ref,i. Da_i (Damköhler) compares the time to cross
the bed with the mass-transfer time 1/k_L. Large Da_i means the solid follows equilibrium closely.

### 3.3 Energy, Eq. (4)

$$\Big(\varepsilon_b \sum_i c_i C_{p,g,i} + \rho_b C_{p,s}\Big)\frac{\partial T}{\partial t}
+ v \sum_i c_i C_{p,g,i}\frac{\partial T}{\partial z}
= \lambda \frac{\partial^2 T}{\partial z^2} + \rho_b \sum_i (-\Delta H_i)\frac{\partial \bar q_i}{\partial t}
- \frac{4 h_w}{D_t}(T - T_w),\qquad T_w = T_{in}$$

Divide by ρ_b C_p,s ΔT_ref / t_ref (the heat capacity of the solid dominates the accumulation):

$$\Big(1 + \sum_i \gamma_i \tilde c_i\Big)\frac{\partial \tilde T}{\partial \tilde t}
+ \tilde v \Big(\sum_i \theta_i \tilde c_i\Big)\frac{\partial \tilde T}{\partial \tilde z}
- \kappa \frac{\partial^2 \tilde T}{\partial \tilde z^2}
- \sum_i H_i \frac{\partial \tilde q_i}{\partial \tilde t}
+ St_w\,\tilde T = 0$$

| Group | Formula | Meaning | Code |
|---|---|---|---|
| γ_i | ε_b C_F C_p,g,i/(ρ_b C_p,s) | gas / solid heat capacity | `gamma` |
| θ_i | v_s C_F C_p,g,i t_ref/(L ρ_b C_p,s) | heat carried by the flow | `conv_T` |
| κ | λ t_ref/(ρ_b C_p,s L²) | axial conduction | `diff_T` |
| H_i | (−ΔH_i) q_ref,i/(C_p,s ΔT_ref) | heat released by adsorption (O(1) by the choice of ΔT_ref) | `heat` |
| St_w | 4 h_w t_ref/(D_t ρ_b C_p,s) | heat lost through the wall | `St_w` |

### 3.4 Overall mole balance, Eq. (5) (gives the velocity)

$$\frac{\partial (v C_T)}{\partial z} = -\varepsilon_b\frac{\partial C_T}{\partial t} - \rho_b \sum_i\frac{\partial \bar q_i}{\partial t} + \varepsilon_b D_L \frac{\partial^2 C_T}{\partial z^2},\qquad C_T = \frac{P}{R T}$$

With C̃_T = C_T/C_F = T_in/T = 1/(1 + ε_T T̃), ε_T = ΔT_ref/T_in, and the same scaling as 3.1:

$$\frac{\partial (\tilde v \tilde C_T)}{\partial \tilde z} + \alpha \frac{\partial \tilde C_T}{\partial \tilde t}
+ \sum_i \beta_i \frac{\partial \tilde q_i}{\partial \tilde t} - \frac{1}{Pe}\frac{\partial^2 \tilde C_T}{\partial \tilde z^2} = 0$$

The derivatives of C̃_T follow from the chain rule:

$$\frac{\partial \tilde C_T}{\partial x} = -\frac{\varepsilon_T}{(1+\varepsilon_T\tilde T)^2}\frac{\partial \tilde T}{\partial x},\qquad
\frac{\partial^2 \tilde C_T}{\partial \tilde z^2} = -\frac{\varepsilon_T}{(1+\varepsilon_T\tilde T)^2}\frac{\partial^2 \tilde T}{\partial \tilde z^2}
+ \frac{2\varepsilon_T^2}{(1+\varepsilon_T\tilde T)^3}\Big(\frac{\partial \tilde T}{\partial \tilde z}\Big)^2 .$$

Why C̃_T comes from the temperature and not from Σ c̃_i: summing Eq. (1) over i gives Eq. (5)
with C_T = Σ c_i, so that version adds no information. What fixes the velocity is the ideal-gas
law at constant pressure, C_T = P/(RT). The PINN therefore uses C̃_T(T̃) in Eq. (5) and adds the
closure residual

$$\sum_i \tilde c_i - \frac{1}{1+\varepsilon_T \tilde T} = 0 .$$

Both terms form `L_global`.

## 4. Boundary and initial conditions, Eq. (6)

Inlet z̃ = 0 (Danckwerts). From ε_b D_L ∂c_i/∂z = v_s (c_i − c_i,F):

$$\frac{1}{Pe}\frac{\partial \tilde c_i}{\partial \tilde z} - (\tilde c_i - y_{i,F}) = 0 .$$

From λ ∂T/∂z = v_s Σ c_i C_p,g,i (T − T_in), dividing by v_s C_F C_p,F ΔT_ref / L with
C_p,F = Σ y_i,F C_p,g,i (molar heat capacity of the feed):

$$\frac{1}{Pe_T}\frac{\partial \tilde T}{\partial \tilde z} - \Big(\sum_i \tilde c_i \frac{C_{p,g,i}}{C_{p,F}}\Big)\tilde T = 0,\qquad Pe_T = \frac{L v_s C_F C_{p,F}}{\lambda}.$$

Outlet z̃ = 1: ∂c̃_i/∂z̃ = 0 and ∂T̃/∂z̃ = 0.

Initial state t̃ = 0: c̃_i = y_i,0 (the gas that fills the bed), q̃_i = q*_i,0/q_ref,i, T̃ = 0, and
ṽ = 1. The last one holds because the initial bed is uniform and in equilibrium, so nothing
changes in time and Eq. (5) gives ∂(v C_T)/∂z = 0, i.e. v = v_s everywhere.

Velocity at the inlet: v(0, t) = v_s, i.e. ṽ(0, t̃) = 1.

Which of these are exact in the network (hard constraints, `models/pinn.py`):

$$\tilde c_i = e^{-\tilde t/\delta}\,\tilde c_{i,0} + \big(1 - e^{-\tilde t/\delta}\big)\,\mathrm{softplus}(N_{c,i}),\qquad
\tilde q_i = e^{-\tilde t/\delta}\,\tilde q_{i,0} + \big(1 - e^{-\tilde t/\delta}\big)\,\mathrm{softplus}(N_{q,i}),$$

$$\tilde T = \big(1 - e^{-\tilde t/\delta}\big) N_T,\qquad \tilde v = 1 + \tilde z\,\big(1 - e^{-\tilde t/\delta}\big) N_v .$$

The weighted average of two non-negative numbers is non-negative, so c̃ and q̃ stay positive and
still match the initial state exactly. (The literal form u_0 + (1 − e^{−t̃/δ}) softplus(N) would force
c̃ ≥ c̃_0, which is wrong for the initial gas, whose concentration falls.) The Danckwerts and
outlet conditions are soft (loss terms).

## 5. Residual normalization

Each residual is divided by its largest coefficient so that all of them are O(1) and the loss
weights start from comparable values:

| Residual | Divided by |
|---|---|
| mass, component i | max(1, α, 1/Pe, β_i) |
| LDF, component i | max(1, Da_i) |
| energy | max(1 + Σγ_i, Σθ_i, κ, ΣH_i, St_w) |
| global | max(1, α, Σβ_i, 1/Pe) |

The LDF division matters: Da can reach ~1800 (k_L = 1 s⁻¹, t_ref = 1800 s).

## 6. Mole balance identity (evaluation)

Integrating Eq. (1) of component i over the bed and over time, from t = 0 until the bed is
back at the feed state:

$$\int_0^\infty \Big[1 - \frac{v(L,t)\,c_i(L,t)}{v_s\,c_{i,F}}\Big]dt = t_{st,i} - t_{init,i},\qquad
t_{init,i} = \frac{L}{v_s\,c_{i,F}}\big(\varepsilon_b c_{i,0} + \rho_b q_{i,0}\big).$$

For a component that is not in the bed at t = 0 (t_init,i = 0) this is the identity of
Section 2.4 (the solver closes it within 0.01%). The extra term is needed for the gas that
fills the bed initially. It does not depend on dispersion, kinetics or velocity variation, so
it is a physics check of the predicted ṽ and c̃ that the MLP cannot provide
(`eval/mass_balance.py`).

## 7. Example: IC1 case (H2/CO2, `configs/cases/ic1_example_h2_co2.json`)

| Quantity | Value |
|---|---|
| C_F | 969.66 mol/m³ |
| c_F (H2, CO2) | 540.27, 429.39 mol/m³ |
| q*_F (H2, CO2) | 0.3934, 0.4973 mol/kg |
| q*_0 (H2, CO2) | 0.6638, 0 mol/kg |
| t_st (H2, CO2) | 39.58, 49.70 s |
| t_ref, t_F | 49.70 s, 99.41 s |
| ΔT_feed (old ΔT_ref) | 33.64 K |
| measured max\|T − T_in\| (MOL, nx = 281) | 4.85 K (field), 4.25 K (outlet) |
| α | 0.451 |
| Pe | 623 |
| Pe_T | 1165 |
| K (H2, CO2) | 0.719, 0.539 |
| Da (H2, CO2) | 25.2, 9.34 |
| γ | 0.031, 0.040 |
| θ | 0.068, 0.088 |
| κ | 6.6e-5 |
| H (with ΔT_feed) | 1.13, 0.84 |
| St_w | 0.98 |
| ε_T (with ΔT_feed) | 0.110 |

High Pe and Pe_T mean sharp concentration and temperature fronts (the reason for the Fourier
features and the residual-based sampling).
