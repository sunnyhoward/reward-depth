# Colleague handoff

The distributable contains source, tests, examples, and built Python artifacts.

Recommended first check:

```bash
python -m pip install -e '.[test]'
pytest
replay-kfac-ewc --help
```

For a real run:

1. Pin the exact frozen model revision.
2. Generate a small replay smoke test and inspect decoded samples.
3. Merge once and retain the library manifest.
4. Run `estimate` first with one projection and `--max-records 10`.
5. Check the printed dense-memory estimate before expanding the target set.
6. Put factor checkpoints on storage with room for atomic replacement.
7. Verify `manifest.json` with the `inspect` command.
8. Confirm that the adapter module names resolve and one anchored training step
   produces nonzero gradients.
9. Calibrate on small perturbations and held-out replay before a long run.

The package intentionally has no hidden cluster launcher or project data
dependency. Model download policy, credentials, job scheduling, preference
loss, checkpointing, and experiment tracking remain with the caller.
