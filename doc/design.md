# Design Principles

## 1. Custom tools; Shared file conventions

It is hard to anticipate future research directions: the most interesting results come from investigations that deviate from common expectations. Shared infrastructure should therefore leave as much freedom in designing new experiments as possible. We do this by encouraging one-off tools for specific projects.

Nonetheless, many kinds of experiments do have to reimplement similar infrastructure and perform similar analyses. We therefore encourage the reuse of well-defined conventions on how to serialize [experiments](./experiment.md) on disk. Custom tools that adhere to those conventions may therefore offer productivity gains by leveraging other tools built by others for unorignal tasks.

## 2. Make Results Deterministic and Replicable 

## 3. Extend, Don't Overwrite, Results


