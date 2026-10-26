"""Run the same bounded CLI as the installed ``cess`` console entry point."""


def main(argv=None):
    from .cli.main import main as cli_main
    return cli_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
