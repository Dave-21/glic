"""Lightweight entry point for generating storyboard PNGs and overlays.

The actual plotting logic lives in `visualize_v2.py`. This wrapper exists so
that downstream scripts can call `python visualize.py --help` without needing
to know which module holds the implementation.
"""

from visualize_v2 import main as visualize_main


def main():
    """Proxy to the newer visualization CLI."""

    visualize_main()


if __name__ == "__main__":
    main()
