"""Compatibility entry point for the linear + RMSNorm scheduler example."""

from __future__ import annotations

from kernels.linear_rmsnorm import validate_linear_rmsnorm_mega_scheduler


def main() -> None:
    validate_linear_rmsnorm_mega_scheduler()
    print("TLE task-graph linear + RMSNorm scheduler validation passed")


if __name__ == "__main__":
    main()
