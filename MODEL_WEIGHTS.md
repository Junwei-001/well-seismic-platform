# 模型权重

仓库包含 FaultSeg 和 Seismic Surface Seg 推理权重。完整清单、文件大小与 SHA-256 位于 `models/manifest.json`。

SurfaceSeg 的 Mask2Former 权重约 1.20 GiB，超过 GitHub 普通文件限制，必须通过 Git LFS 提交：

```bash
git lfs install
git lfs pull
python tools/verify_release.py
```

如果自检提示文件为 LFS pointer，说明权重实体尚未下载；执行 `git lfs pull` 后重试。普通 GitHub 账户的 LFS 存储和流量额度可能不足，公开发布前应核对账户额度，或将权重放到 GitHub Release / Hugging Face 并在配置中提供可校验的下载方式。

