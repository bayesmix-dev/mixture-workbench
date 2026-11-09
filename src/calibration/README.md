# Prior and geometry calibration

This directory contains the original prior-occupancy, distance, repulsion and Strauss calibration procedures. These select or compare model settings; the prior laws themselves live under [priors](../priors/README.md), and sampler state lives under [models](../models/README.md).

Calibration is an explicit preparation step. Record the input data or reference settings, selected parameters and numerical error or tail bounds supplied by the procedure. A model initialized from empirical moments must retain that provenance in the run metadata.

Existing calibration functions target particular Gaussian MFM, spectral DPP or Strauss constructions. Reusing a function for a new model requires checking its parameterization and defining the quantity being matched. Changing calibrated geometry may also require rebuilding model caches and any replica-specific workspaces.

Read [docs/targets](../../docs/targets/README.md) and [add-model](../../recipes/add-model.md) before adapting a calibration. The CLI uses only explicitly implemented settings and records resolved parameters through [cli](../cli/README.md).
