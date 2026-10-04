"""Універсальне визначення кількості потоків CPU для TTS-інференсу.

Правила вибору (в порядку пріоритету):
  1. Якщо задано env TTS_THREADS -> використовується це значення (ручний вибір).
  2. Інакше авто: int(os.cpu_count() * 0.8) - 80% потоків хоста (переносимість
     на інші процесори без зміни коду).
  3. Якщо контейнер має cgroup-квоту CPU, меншу за авто -> обрізка до квоти
     (cpu.max для cgroup v2, cpu.cfs_quota_us/period_us для v1), щоб не
     отримувати throttling.

apply_threads() викликається ДО завантаження torch: виставляє
OMP_NUM_THREADS/MKL_NUM_THREADS, потім torch.set_num_threads/set_num_interop_threads.
"""
import os


def _cgroup_cpu_quota():
    """Квота CPU контейнера (скільки ядер дозволено) або None."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
            if quota != "max":
                return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read().strip())
        if q > 0 and p > 0:
            return q / p
    except (OSError, ValueError):
        pass
    return None


def resolve_threads():
    env = os.environ.get("TTS_THREADS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    host = os.cpu_count() or 1
    threads = max(1, int(host * 0.8))
    quota = _cgroup_cpu_quota()
    if quota is not None:
        threads = min(threads, max(1, int(quota)))
    return threads


def apply_threads():
    """Виставляє потоки для torch/OMP і повертає їх кількість."""
    t = resolve_threads()
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["MKL_NUM_THREADS"] = str(t)
    import torch
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # interop вже ініціалізовано
    torch.set_num_threads(t)
    env = os.environ.get("TTS_THREADS", "").strip()
    mode = "manual TTS_THREADS=" + env if env else "auto 80%"
    host = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    qtxt = "-" if quota is None else ("%.1f" % quota)
    print("[threads] threads=%d (%s: host=%d, quota=%s)" % (t, mode, host, qtxt), flush=True)
    return t
