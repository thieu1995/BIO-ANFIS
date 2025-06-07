

# Version 1.1.0

### Dependencies
- Updated `mealpy` from version **3.0.1** to **3.0.2**.
- Updated `permetrics` from version **1.5.0** to **2.0.0**.
- Updated `requirements.txt` accordingly.

### Metadata and Configuration
- Updated metadata files: `CITATION.cff`, `MANIFEST.in`, and `requirements.txt`.

### Codebase Improvements
- Updated GitHub workflows.
- Refreshed example scripts and requirement files.

### Bug Fixes
- Fixed missing `loss_train` attribute in classic and gradient-based models.
- Fixed issue with device assignment (CPU/GPU) for classic and gradient-based models.

### Refactoring
- Moved parameters (`lb`, `ub`, `mode`, `n_workers`, `termination`) from the `fit()` method to the model constructor (`__init__()`) in bio-inspired models.

---------------------------------------------------------------------------------------

# Version 1.0.0

The first official release of X-ANFIS includes:

+ Add infors (CODE_OF_CONDUCT.md, MANIFEST.in, LICENSE, requirements.txt)
+ Add helpers modules (membership_family, metric_util, scaler_util, validator, preprocessor)
+ Add wrapper for different ANFIS models (BaseAnfis, BaseClassicAnfis, BaseGdAnfis, BaseBioAnfis)
+ Add wrapper for different ANFIS models (AnfisRegressor, AnfisClassifier, GdAnfisRegressor, GdAnfisClassifier, BioAnfisRegressor, BioAnfisClassifier)
+ Add publish workflow
+ Add examples and tests folders
