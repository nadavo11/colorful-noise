# FLUX SeaCache DP Shortcuts

Verdict: **PROCEED_WEAK**

The expanded HTML report includes a figure for each experiment, SeaCache replication visuals, DP statistics, perceptual metrics, h/cost correlation plots, and a consecutive-shortcut histogram. The h tensor is the first-block norm1 modulated image-token input used by SeaCache; middle steps are SEA-filtered as in the official decision path. The cache-aware DP cost uses compact h-summary RMSE for tractability and should be treated as a proxy until validated online. Offline DP images are decoded from reconstructed latents, not executable sampler runs.
