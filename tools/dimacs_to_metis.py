# Converts a DIMACS-graph into the METIS format
# This is a modified version of the original script from the KaMIS repository
# https://github.com/KarlsruheMIS/KaMIS/
import re
import logging


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split("(\d+)", text)]


def convert(dimacs):
    number_nodes = 0
    number_edges = 0
    edges_counted = 0
    adjacency = []

    for line in dimacs.splitlines():
        args = line.strip().split()
        if "cf" in args or "co" in args or "c" in args:
            continue
        if len(args) == 4:
            type, source, target, _ = args
        else:
            type, source, target = args
        if type == "p":
            number_nodes = source
            number_edges = target
            logging.debug(
                "Given dimacs graph has "
                + number_nodes
                + " nodes and "
                + number_edges
                + " edges"
            )
            adjacency = [[] for _ in range(0, int(number_nodes) + 1)]
        elif type == "e" or type == "a":
            edge_added = False
            if not target in adjacency[int(source)]:
                adjacency[int(source)].append(target)
                edge_added = True
            if not source in adjacency[int(target)]:
                adjacency[int(target)].append(source)
                edge_added = True
            if edge_added:
                edges_counted += 1
        else:
            logging.warning("Could not read line.")

    adjacency[0].append(number_nodes)
    # adjacency[0].append(number_edges)
    adjacency[0].append(str(edges_counted))

    metis = ""

    node = 0
    for neighbors in adjacency:
        if node != 0:
            neighbors.sort(key=natural_keys)
        if not neighbors:
            metis += " "
        else:
            metis += " ".join(neighbors)
        metis += "\n"
        node += 1

    logging.debug("Finished converting graph to metis format")
    return metis
