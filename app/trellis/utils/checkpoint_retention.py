# 2026-09-22：按完整 checkpoint 套组保留最近节点与里程碑，避免历史权重无限累积。
from pathlib import Path
import re


def prune_checkpoints(directory, model_names, ema_rates, keep_last=0, milestones=()):
    """Prune only complete known bundles. Disabled by default; leave partial files alone."""
    if keep_last <= 0:
        return []
    root = Path(directory)
    prefixes = [name for name in model_names]
    prefixes += [f'{name}_ema{rate}' for name in model_names for rate in ema_rates]
    prefixes += ['misc']
    bundles = {}
    for misc in root.glob('misc_step*.pt'):
        match = re.fullmatch(r'misc_step(\d{7,})\.pt', misc.name)
        if not match:
            continue
        step = int(match.group(1))
        files = [root / f'{prefix}_step{step:07d}.pt' for prefix in prefixes]
        if all(p.is_file() and not p.is_symlink() and p.stat().st_size > 0 for p in files):
            bundles[step] = files
    protected = set(sorted(bundles)[-keep_last:]) | set(milestones)
    deleted = []
    for step in sorted(bundles):
        if step in protected:
            continue
        # latest 按 misc 发现节点，先移除 misc，避免中途停止留下可发现但缺件的节点。
        for path in reversed(bundles[step]):
            path.unlink()
            deleted.append(str(path))
    return deleted
