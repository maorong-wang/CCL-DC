"""Configuration file for results generation.
This file should contain the list of parameters to include in the generated latex table.
Example:
params_to_compare = {
    "prefix",
    "dataset",
    "mem_size"
}
Parameters are found by parsing params_used.json file. Value for every paremeters will be found, even useless ones.
Please be careful that the parameters you ask for are used by the method.
Example: STAM method will have a mem_size parameters but has no use for it.

To find the list of possible parameter values refer to config/parser.py
"""
params_to_compare = {
    "dataset",
    "mem_size",
    "gram_th"
}