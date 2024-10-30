import argparse
import importlib.util
import os
import sys


def load_config(config_file):
    """
    Loads configuration parameters from a specified Python file.

    Args:
      config_file (str): Path to the configuration file.

    Returns:
      module: The imported module containing configuration variables.

    Raises:
      ImportError: If the specified config file cannot be imported.
    """
    spec = importlib.util.spec_from_file_location("constants", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_config_path(args):
    """
    Determines the path to the configuration file based on arguments.

    Args:
      args (argparse.Namespace): Parsed command-line arguments.

    Returns:
      str: Path to the configuration file.
    """
    if args.config:
        return args.config
    else:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, "constants.py")
        if os.path.exists(config_path):
            return config_path
        else:
            raise FileNotFoundError(
          "No config.py found in script directory and -c flag not provided."
      )

def parse_args():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument(
        "-c", "--config", help="Path to the configuration file (optional)."
    )
    return parser.parse_args()


def load():
    """
    Parses command-line arguments, determines the configuration file path,
    and loads the configuration.

    Returns:
      module: The imported module containing configuration variables.
    """
    args = parse_args()
    try:
        config_path = get_config_path(args)
        config = load_config(config_path)
        return config
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
