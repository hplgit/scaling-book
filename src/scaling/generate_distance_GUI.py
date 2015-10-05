from parampool.generator.flask import generate
from distance import distance_unit2, define_input

generate(distance_unit2, pool_function=define_input, MathJax=True)
