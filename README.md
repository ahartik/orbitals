# 3D Hydrogen orbital visualization

Run in your browser [HERE](https://ahartik.github.io/orbitals).

Renders hydrogen orbital wavefunctions by raymarching.
Surface is drawn at $|\psi|^2=L$ for given "surface limit" L.
Surface color depends the argument of the wavefunction $\arg(\psi)$.

Features:
* Orbitals up to (and including) N=8.
* Real and complex orbitals.
* Option to render only half of the orbital ("hide y<0") to better see what's inside.
* Visualizes $\psi$ as a "cloud" when $|\psi|^2 < L$.
* Option to reduce rendering resolution to work on low-performance machines ("Render scale")
* Pan using mouse or touch (mobile).
* Zoom using scroll wheel.

Missing features / known bugs:
* Optimize more to achieve smooth experience on mobile without resolution scaling.
* Would be nice to visualize axis directions and some markers to show orbital size in units of Bohr radii.
* Real orbitals are "too small" (don't integrate up to 1)
* Zoom on mobile devices (multitouch).
