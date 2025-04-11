
    SUPPORT_MEMBERSHIP_CLASSES = {
        "Gaussian": "GaussianMembership",
        "Trapezoidal": "TrapezoidalMembership",
        "Triangular": "TriangularMembership",
        "Sigmoid": "SigmoidMembership",
        "Bell": "BellMembership",
        "PiShaped": "PiShapedMembership",
        "SShaped": "SShapedMembership",
        "GBell": "GBellMembership",
        "ZShaped": "ZShapedMembership",
        "Linear": "LinearMembership",
    }

# Version 1.1.0

+ Add `MhaMlpRegressor` and `MhaMlpClassifier` classes
+ Add docs folder and document website
+ Update examples and tests folders
+ Add zenodo DOI, CITATION.cff

----------------------------------------------------------------------------------------

# Version 1.0.1

+ Add infors (CODE_OF_CONDUCT.md, MANIFEST.in, LICENSE, requirements.txt)
+ Add helpers modules (act_util, metric_util, scaler_util, validator, preprocessor)
+ Add MlpRegressor and MlpClassifier classes (Based on Pytorch and Skorch)
+ Add publish workflow
+ Add examples and tests folders
