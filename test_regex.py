#!/usr/bin/env python3
import re

LIST_PATTERN = re.compile(r'^\s*(\d+\.|\d+\)|\d+\-|[a-zA-Z]\.|[a-zA-Z]\)|[a-zA-Z]\-|\•|\-|\*)\s')

test_strings = [
    "1. Select a simple open Brayton–Joule thermodynamic cycle",
    "2. Adopt a turbine-inlet temperature smaller than 1000 K",
    "3. Choose a single-shaft configuration",
    "1. Introduction and background",
    "2. Design specifications",
    "E. Benini, S. Giacometti / Applied Energy 84 (2007) 1102–1116",
]

for s in test_strings:
    match = LIST_PATTERN.match(s.strip())
    if match:
        print(f"MATCH: '{s}' -> '{match.group(0)}'")
    else:
        print(f"NO MATCH: '{s}'")