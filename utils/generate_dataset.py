"""Dataset utility entry point.

No synthetic URLs are generated: this command validates and canonicalizes a
user-provided labeled CSV so training data remains explicitly sourced.
"""

from build_features import main


if __name__ == "__main__":
    main()
