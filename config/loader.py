"""Loads config/config.yaml. Every threshold/path lives there, not in code."""

import os

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)
