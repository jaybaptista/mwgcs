# gchords

## the `Gchords` object

The `Gchords` object contains all the infall modeling methods to perform globular cluster (GC) formation and evolution.

The (main) classes that generate the GC population are:
- `Interface`: a generic interface to the simulation suite to determine the infall properties of subhalos and their particle tracking data.
- `GCHaloModel`: a class that handles the generation of GC populations based on halo (or stellar) mass. It is largely defined by the following subclasses: 
    - `OccupationModel`: is a subhalo massive enough to host a GC system?
    - `GCSMassModel`: what is the infall mass of a GC system?
    - `GCLuminosityFunction`: what is the luminosity distribution of the GC systems?

There are additional classes that facilitate the evolution of the GC systems:
- `Potential`: a class that uses the particle tracking data in the `Interface` to produce a time-evolving potential and evaluate tidal fields.
- `MassLossModel`: a class that defines how a cluster loses mass.

See `examples/example_smw.py` for how these components are used to:
(1) generates an accreted GC population based on infall parameters
(2) constructs a time-evolving potential
(3) evaluates the tidal mass loss of the GC population