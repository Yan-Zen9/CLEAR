"""
Counterfactual generation (flow matching) for MedDiME.

Modules:
    config     - run configuration (env-driven) + print/save helpers
    classifier - unified classifier interface (bundle, normalize, target prob)
    saliency   - Grad-CAM attribution; CF_SALIENCY_MODE selects the support signal
    core       - adaptive masking + classifier guidance + GuidedFlow sampler
    loading    - load velocity model / classifier / VAE / test set + I/O helpers
    generate   - CLI entry point

Note: `cf.config` runs env-driven setup (and prints) on import; import it
explicitly where needed rather than from this package __init__.
"""
