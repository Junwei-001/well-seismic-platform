"""Allow ``python -m minimal_sgy`` to invoke the command-line interface."""

from .infer import main


if __name__ == "__main__":
    main()
