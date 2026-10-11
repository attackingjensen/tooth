# C80 → TRELLIS crown generation (20261011)

C80 SurfaceCrownNet v3 old-only step12320 → TRELLIS SLAT ControlNet raw55k → mesh decoder.
Research prototype; package integrity does not establish clinical quality.

## Package layout

GitHub contains inference code, configs, web demo and dependency source. Hugging Face contains the same code plus the two custom trained weights (original checkpoint bytes and relative paths preserved). No patient scans, reference fixtures, web job history, local credentials or service logs are included.

## Deployment

1. Download the full Hugging Face repository, or copy its `models/` and `app/runs/` into the GitHub checkout. The GitHub `models/dinov2/hub/` source must also remain present.
2. Follow `ENVIRONMENT.md` to create the two Python environments.
3. Copy `env.example` to `env.local` and edit interpreter paths and GPU settings.
4. Run `python verify.py` to check release file hashes. Run `python verify.py --cpu-only` in the inference environment to additionally strictly load C80.
5. Run `./start.sh local`; upload your own jaw scan and select crop and FDI tooth position.

See `ARCHITECTURE.md` and `README_DEPLOYMENT_ORIGINAL.md` for inference conventions and CLI examples. The original deployment README and `VALIDATION_ORIGINAL.md` describe the source package's historical tests; scan fixtures and result files referenced there are intentionally absent from this public release. The release verifier checks integrity and optionally C80 loading only; test full GPU inference on the target machine.

## Provenance and licenses

The C80 and SLAT checkpoint hashes are recorded in `release_manifest.json`. Third-party licenses present in the source package are retained. This release does not invent a license for the custom code or checkpoint weights; their reuse terms must be specified by the owner. TRELLIS, DINOv2 and dependency components remain subject to their upstream terms.

## Released checkpoints and external dependencies

- Stage 1: `models/c80/c80_v3_step12320.pt` (C80 v3 old-only step12320).
- Stage 2: `app/runs/crown_slat_full/ckpts/denoiser_step0055000.pt` (SLAT ControlNet raw55k).

Only these two custom checkpoints are uploaded. Before inference obtain the following upstream weights and place them at exactly these paths:

- DINOv2: `models/dinov2/dinov2_vitl14_reg4_pretrain.pth` ([official repository](https://github.com/facebookresearch/dinov2)).
- TRELLIS encoder: `models/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors`.
- TRELLIS mesh decoder: `models/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors` ([official model repository](https://huggingface.co/microsoft/TRELLIS-image-large)).

The TRELLIS JSON configs remain included. File verification checks published files; it does not imply that external weights are installed.

Code: https://github.com/attackingjensen/tooth
Checkpoints: https://huggingface.co/zhaxie/tooth

## 2026-10-11 update

C80 is now v3 old-only step12320, trained with full0.7 baseline, scale jitter and center offsets. SLAT baseline raw55k is unchanged.

The default crop recipe is `v3_aligned`: explicitly supply the estimated crown bounding-box center and longest edge d; cube side L=d/0.7. Input axes must already follow the training convention. The default does not apply PCA rotation or normal-based center displacement. The web UI requires the diameter and allows center correction. Re-estimate and preview old tasks; old full0.9 saved candidates are expanded by 9/7. This is not automatic FDI tooth localization. Stage 2 retains its original full0.9 training recipe.

```bash
python app/prepare_jaw_frame.py --jaw aligned.stl --center X Y Z --diameter 9 --out_dir prepared
```

See `WEIGHT_UPDATE.json` for checkpoint provenance and `VALIDATION.md` for the latest package validation scope. Historical deployment instructions refer to fixtures not included in this public release.
