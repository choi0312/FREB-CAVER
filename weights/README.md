# Released adapter

The adapter is hosted as a GitHub Release asset because its 132,318,080-byte
size exceeds GitHub's normal 100 MiB repository-object limit.

```bash
gh release download v0.1.0 \
  --repo choi0312/FREB-CAVER \
  --pattern "native_deep_residual*" \
  --dir checkpoints/FREB-CAVER
```

Expected files:

- `native_deep_residual.safetensors`
- `native_deep_residual_identity.json`

Expected weights SHA-256:

```text
ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed
```

The identity JSON is versioned in Git so architecture settings can be inspected
without downloading the tensor file.
