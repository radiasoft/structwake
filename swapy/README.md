# swapy
`swapy` (***s***tructure-based ***w***akefield ***a***cceleration with ***Py***thon) is a package for modeling particle acceleration via structure-based wakefields. 

## Installation

To install this package, first download a copy of the source code. Once downloaded, you can install the package by running the
following command:

```python3 -m pip install ./swapy```

from one directory above the top-level `swapy` folder containing *pyproject.toml*. If installation was successful, you should now be able to execute an import (`import swapy`) from a Python script or interactive editor.

## Documentation

Documentation for this package can be found on the wiki of this repository (coming soon).

## References

A significant portion of the models for wakefields in axisymmetric, dielectric-lined waveguides 
was adapted from a codebase developed by P. Piot and the Advanced Accelerator R&D group at
Northern Illinois University. A repository containing this work can be found at:
* https://github.com/NIUaard/DiWakeCyl/

The codebase referenced above and theoretical background used in this package are based on the 
treatment provided by Ng (1990):
* K-Y. Ng, "Wake fields in a dielectric-lined waveguide," Phys. Rev. D 42 (1990)
  * https://doi.org/10.1103/PhysRevD.42.1819
