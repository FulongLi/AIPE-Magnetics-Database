# Large-Signal Properties in MnZn Ferrites: Code and Data

## Introduction

This repository contains the **datasets** and **code** related to the following **paper**:
* **Characterization and Impact of Large-Signal Dielectric Properties in MnZn Ferrites**
* **Thomas Guillod, William V. R. Roberts, and Charles R. Sullivan**
* **IEEE APEC 2024**
* [**DOI** (IEEE copyright)](https://doi.org/10.1109/APEC48139.2024.10509366)
* [**Paper** (IEEE copyright)](paper.pdf)
* [**Slides** (CC BY-ND 4.0)](slides.pdf)

This code simulates the **magnetic-dielectric effects** in a infinitely long **cylindrical core**:
* The (average) flux density through the cross-section is imposed (excitation).
* The material parameters can be amplitude-dependent (locally linearized).

Two different **solvers** are available:
* **Quasi-static** approximation (no wave propagation and spatially independent material parameters).
* **Full wave** solution (with wave propagation and spatially dependent material parameters).

## Main Files

* [paper.pdf](paper.pdf) - PDF of the paper (IEEE copyright) 
* [slides.pdf](slides.pdf) - PDF of the slides (CC BY-ND 4.0) 
* [get_param.m](get_param.m) - Definition of the solver parameters
* [run_single.m](run_single.m) - Solve the electromagnetic problem in a core with a given cross-section 
* [run_sweep.m](run_sweep.m) - Solve the electromagnetic problem in cores with different cross-sections

## Datasets

* [N87_eps_small](dataset/N87_eps_small) - EPCOS TDK N87 - Small-Signal Permitivitty
* [N87_eps_large](dataset/N87_eps_large) - EPCOS TDK N87 - Large-Signal Permitivitty
* [N87_eps_temperature](dataset/N87_eps_temperature) - EPCOS TDK N87 - Temperature Permitivitty
* [N87_mu_small](dataset/N87_mu_small) - EPCOS TDK N87 - Small-Signal Permeability
* [N87_mu_large](dataset/N87_mu_large) - EPCOS TDK N87 - Large-Signal Permeability

## Compatibility

* Tested with MATLAB R2023a.
* The `pde_toolbox` is required.

## Author

* **Thomas Guillod**
* Email: guillod@otvam.ch
* Website: https://otvam.ch

## Credits

This code was created at **Dartmouth College** by the research group of **Prof. Sullivan**:
* Dartmouth College, NH, USA: https://dartmouth.edu
* Dartmouth Engineering: https://engineering.dartmouth.edu
* PMIC: https://pmic.engineering.dartmouth.edu

## License

This project is licensed under the **MIT License**, see [LICENSE.md](LICENSE.md).
