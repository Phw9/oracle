# Oracle Local Model

The GGUF models are downloaded by `./build.sh` only when no usable `*.gguf`
model already exists under this `models/` directory. `./run.sh` also reuses an
existing repo model before attempting any download.

## Default Runtime Model

- Source: `unsloth/gemma-3-1b-it-GGUF`
- File: `gemma-3-1b-it-Q4_0.gguf`
- URL: `https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_0.gguf`
- Size: `721,918,496` bytes
- SHA256: `27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e`
- License: Gemma
- Path: `models/gemma-3-1b-it-Q4_0.gguf`

The model files are not committed to this repository. `*.tmp` files are used for
resumable downloads, and completed files are moved into place after SHA256
verification. If a matching GGUF file is already present, the scripts verify the
known packaged hashes and skip downloading. The default app configuration treats
these as text-first models and sends capture metadata instead of image bytes.
Use a compatible multimodal GGUF plus projector before setting
`ORACLE_FACE_LLM_SEND_IMAGE=1`.
