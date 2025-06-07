

# Version 1.1.0

+ Update dependency versions in requirements.txt (mealpy from 3.0.1 to 3.0.2, permetrics from 1.5.0 to 2.0.0)
+ Update infors (CITATION.cff, MANIFEST.in, requirements.txt)
+ Update workflows, requirements, examples.
+ Fix bug missing loss_train in classic and gd-based models.
+ Fix bug put classic models and gd-based models to device (gpu/cpu)
+ Move parameters from fit() to init() in bio-based models include lb, ub, mode, n_workers, termination.

---------------------------------------------------------------------------------------

# Version 1.0.0

The first official release of X-ANFIS includes:

+ Add infors (CODE_OF_CONDUCT.md, MANIFEST.in, LICENSE, requirements.txt)
+ Add helpers modules (membership_family, metric_util, scaler_util, validator, preprocessor)
+ Add wrapper for different ANFIS models (BaseAnfis, BaseClassicAnfis, BaseGdAnfis, BaseBioAnfis)
+ Add wrapper for different ANFIS models (AnfisRegressor, AnfisClassifier, GdAnfisRegressor, GdAnfisClassifier, BioAnfisRegressor, BioAnfisClassifier)
+ Add publish workflow
+ Add examples and tests folders
