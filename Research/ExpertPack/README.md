# Routide Expert Pack

Routide converts the pinned MLX SafeTensors checkpoint into a storage-addressable
format without loading or dequantizing the whole model.

The output layout is:

```text
manifest.json
resident.bin
experts/
  layer-000.bin
  ...
```

Each routed expert occupies a fixed-stride block aligned to 64 KiB. For the
paper's pinned Qwen3.6 checkpoint, one expert payload is 1,769,472 bytes, or
exactly 27 × 64 KiB.

## Convert a checkpoint

```bash
cd Research/ExpertPack
PYTHONPATH=. python3 -m routide_pack.packer \
  /path/to/Qwen3.6-35B-A3B-4bit \
  /path/to/routide-qwen36-pack \
  --model-id mlx-community/Qwen3.6-35B-A3B-4bit \
  --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46
```

The source checkpoint and output pack together require roughly 41 GB before
filesystem overhead. Reserve additional free space on the target device.

## Validate

```bash
PYTHONPATH=. python3 -m routide_pack.reader /path/to/routide-qwen36-pack --verify-hashes
python3 -m unittest discover -s tests -v
```

The packer is bounded-memory and the synthetic tests do not download model
weights. The generated model files are excluded from Git.
