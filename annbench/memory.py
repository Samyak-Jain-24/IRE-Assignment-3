"""Keep the benchmark's memory resident while it is being timed.

On a machine with little free RAM, Windows trims the working set of a busy process and
compresses or pages out what it trims. A scan over an index then pays for page faults instead of
distance computations. pin_working_set() sets a hard minimum working set so the process's pages
are not trimmed; page_faults() lets every timed pass record how many faults it took.
"""
import ctypes
import sys

import psutil

_PROC = psutil.Process()


def pin_working_set(min_bytes: int) -> bool:
    """Ask Windows to keep at least min_bytes of this process resident. No-op elsewhere."""
    if sys.platform != "win32":
        return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32]
    k32.SetProcessWorkingSetSizeEx.restype = ctypes.c_int
    QUOTA_LIMITS_HARDWS_MIN_ENABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x1, 0x8
    ok = k32.SetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), min_bytes, max(min_bytes * 2, 1 << 30),
                                        QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)
    if not ok:
        print(f"[memory] could not pin {min_bytes / 2**30:.1f} GiB working set (error {ctypes.get_last_error()})",
              flush=True)
    return bool(ok)


def page_faults() -> int:
    """Cumulative page faults of this process (Windows; 0 where psutil does not report them)."""
    return int(getattr(_PROC.memory_info(), "num_page_faults", 0))


def rss_mib() -> float:
    return _PROC.memory_info().rss / 2**20


def on_ac_power() -> bool:
    """True if mains-powered or the machine has no battery."""
    try:
        bat = psutil.sensors_battery()
    except Exception:
        return True
    return bat is None or bool(bat.power_plugged)
