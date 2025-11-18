#!/usr/bin/env python3
"""Entry point for pruning outdated daily renders from S3."""

from fog.daily_render_cleanup import main


if __name__ == "__main__":
    raise SystemExit(main())

