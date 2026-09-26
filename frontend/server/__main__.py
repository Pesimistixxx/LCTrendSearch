"""Run from the repository root: python -m frontend.server."""

import argparse

from .app import serve


def main():
    parser = argparse.ArgumentParser(
        description="Локальный интерфейс загрузки материалов"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5188)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
