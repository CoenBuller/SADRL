## Submodule 

The "socialjax" folder is a sub repository. Dont change anything in that repository, only use code from it and build algorithms or experiments around it.uv 
## Reputation patch (branch `reputation`)

`patches/0001-Add-opt-in-reputation-observation-channel.patch` adds an opt-in reputation observation channel to SocialJax (envs, `socialjax.make`, configs, `train.py`, tests). It applies cleanly to upstream SocialJax (`c64f1aa`). Apply it to your local `socialjax` clone:

    git -C socialjax am ../patches/0001-Add-opt-in-reputation-observation-channel.patch

Then enable it per run, e.g. `uv run python -m socialjax.algorithms.train --algo IPPO --env pd_arena REPUTATION=True REPUTATION_LAMBDA=0.95`. See `coen.md` for the design.
