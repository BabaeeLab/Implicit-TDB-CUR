# Implicit TDB-CUR

Code and data accompanying the manuscript titled "CUR for Implicit Time Integration of Random Partial Differential Equations on Low-Rank Matrix Manifolds", authored by Mohammad Hossein Naderi, Sara Akhavan, and Hessam Babaee.


# Abstract
Dynamical low-rank approximation allows for solving large-scale matrix differential equations (MDEs) with significantly fewer degrees of freedom and has been applied to a growing number of applications. However, most existing techniques rely on explicit time integration schemes. In this work, we introduce a cost-effective Newton's method for the implicit time integration of stiff, nonlinear MDEs on low-rank matrix manifolds. Our methodology is focused on MDEs resulting from the discretization of random partial differential equations (PDEs). Cost-effectiveness is achieved by solving the MDE at the minimum number of entries required for a rank $r$ approximation. We present a novel CUR low-rank approximation that requires solving the parametric PDE at $r$ strategically selected parameters and $\mathcal{O}(r)$ grid points using Newton's method. The selected random samples and grid points adaptively vary over time and are chosen using the discrete empirical interpolation method or similar techniques. The proposed methodology is developed for high-order implicit multistep and Runge-Kutta schemes and incorporates rank adaptivity, allowing for dynamic rank adjustment over time to control error. Several analytical and PDE examples, including the stochastic Burgers' and Gray-Scott equations, demonstrate the accuracy and efficiency of the presented methodology.


# Citation
@misc{naderi2024curimplicittimeintegration,
      title={CUR for Implicit Time Integration of Random Partial Differential Equations on Low-Rank Matrix Manifolds}, 
      author={Mohammad Hossein Naderi and Sara Akhavan and Hessam Babaee},
      year={2024},
      eprint={2408.16591},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2408.16591}, 
}


The repository contains all the necessary code and data to reproduce the results in the paper. 
