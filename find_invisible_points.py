from cgshop2023_pyutils import io
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point, LineString

from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
import pytest
import random
import logging
import time
import os
import json
import argparse
import pandas as pd
import multiprocessing as mp

random.seed(42)

VERSION = "0.2.2"
DEBUG = False
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


class CsvLogger:
    def __init__(self, output_folder: str):
        self.df = pd.DataFrame(
            columns=[
                "instance",
                "num_nodes",
                "num_holes",
                "shapely_load_time",
                "sampling_method",
                "machine",
                "version",
                "sampling_time",
                "graph_creation_time",
                "graph_nodes",
                "graph_edges",
                "graph_solving_time",
                "num_invisible_points",
                "total_time",
            ]
        )
        self.current_instance = None
        self.output_folder = output_folder

    def log(self, column: str, value):
        self.df.loc[self.current_instance, column] = value

    def set_instance(self, instance: str):
        self.current_instance = instance
        self.df.loc[self.current_instance, "instance"] = instance
        self.df.loc[self.current_instance, "machine"] = os.uname()[1]
        self.df.loc[self.current_instance, "version"] = VERSION

    def save(self):
        self.df.to_csv(
            self.output_folder
            + "/log"
            + datetime.now().strftime("%Y%m%d%H%M%S")
            + ".csv",
            index=False,
        )

    def remove_instance(self):
        self.df.drop(self.current_instance, inplace=True)
        self.current_instance = None


def load_polygon(filename: str, csv_logger: CsvLogger = None) -> Polygon:
    """
    Loads a polygons from a filename
    """
    start_time = time.time()
    instance = io.read_instance(filename)
    logging.debug(f"Loaded instance {filename}")
    p = Polygon(
        [(p["x"], p["y"]) for p in tqdm(instance["outer_boundary"], disable=not DEBUG)]
    )
    for hole in tqdm(instance["holes"], disable=not DEBUG):
        p = p.difference(Polygon([(p["x"], p["y"]) for p in hole]))
    if csv_logger:
        csv_logger.log("shapely_load_time", time.time() - start_time)
        csv_logger.log("num_nodes", len(p.exterior.coords))
        csv_logger.log("num_holes", len(p.interiors))
    logging.debug(
        f"Created polygon with shapely in {time.time() - start_time}s. There are {len(p.interiors)} holes, {len(p.exterior.coords)} nodes in the outer boundary."
    )
    return p


def sample_alternating_holes_points(p: Polygon) -> set[Point]:
    """
    Samples points from the polygon and its holes, alternating between nodes in the polygon and nodes in the holes.
    TODO: explain better
    """
    points = set()
    for hole in p.interiors:
        # centroid = Polygon(hole).centroid
        for i in range(0, len(hole.coords), 2):
            point = Point(hole.coords[i])
            # move the point a little towards the centroid
            # point = Point(point.x - (centroid.x - point.x) * 0.1, point.y - (centroid.y - point.y) * 0.1)
            points.add(point)
    # centroid = p.centroid
    for i in range(0, len(p.exterior.coords), 2):
        point = Point(p.exterior.coords[i])
        # move the point a little towards the centroid
        # point = Point(point.x + (centroid.x - point.x) * 0.1, point.y + (centroid.y - point.y) * 0.1)
        points.add(point)
    return points


def sample_random_points(p: Polygon, num_samples: int) -> set[Point]:
    """
    Samples points inside the polygon uniformly at random.
    """
    points = set()
    while len(points) < num_samples:
        x = random.uniform(p.bounds[0], p.bounds[2])
        y = random.uniform(p.bounds[1], p.bounds[3])
        point = Point(x, y)
        if p.contains(point):
            points.add(point)
    return points


def sample_mid_points(p: Polygon) -> set[Point]:
    """
    Samples points that are in the middle of the hole's edges and the outer boundary edges.
    """
    points = set()
    for hole in p.interiors:
        for i in range(len(hole.coords) - 1):
            point = Point(
                (hole.coords[i][0] + hole.coords[i + 1][0]) / 2,
                (hole.coords[i][1] + hole.coords[i + 1][1]) / 2,
            )
            points.add(point)
    for i in range(len(p.exterior.coords) - 1):
        point = Point(
            (p.exterior.coords[i][0] + p.exterior.coords[i + 1][0]) / 2,
            (p.exterior.coords[i][1] + p.exterior.coords[i + 1][1]) / 2,
        )
        points.add(point)
    return points


def can_see(p: Polygon, q: Point, r: Point) -> bool:
    """
    Checks if point q can see point r in the polygon p (including holes)
    """
    l = LineString([q, r])
    if l.crosses(p.exterior):
        return False
    # FIXME: no need to iterate through all of them, have preselection
    # This is a big bottleneck at the moment. Should be optimized for midpoint
    # Ideally, we could already compute whether two points are visible from each other
    # when we sample them at the midpoint
    for hole in p.interiors:
        # FIXME: this doesn't work for two points on the same hole
        if l.crosses(hole) or l.within(hole):
            return False
    return True


def find_invisible_points_within_set(
    p: Polygon, points: set, can_see: callable = can_see, csv_logger: CsvLogger = None
) -> set[Point]:
    """
    Finds a set of points that are mutually invisible with each other within the polygon given a set of points.
    """
    points = list(points)

    start_time = time.time()
    # create a graph where two points are connected if they can see each other
    tot_nodes, tot_edges = 1, 0
    dimacs_graph = ""

    logging.info(f"Using {mp.cpu_count()} cores")
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(
        can_see,
        [
            (p, points[i], points[j])
            for i in range(len(points) - 1)
            for j in range(i + 1, len(points))
        ],
    )
    results = results[::-1]
    pool.close()
    # TODO: this can be optimized
    for i in tqdm(range(len(points) - 1)):
        for j in range(i + 1, len(points)):
            if results.pop():
                dimacs_graph += f"e {i + 1} {j + 1}\n"
                tot_edges += 1
        tot_nodes += 1
    dimacs_graph = f"p {tot_nodes} {tot_edges}\n" + dimacs_graph
    if csv_logger:
        csv_logger.log("graph_creation_time", time.time() - start_time)
        csv_logger.log("graph_nodes", tot_nodes)
        csv_logger.log("graph_edges", tot_edges)
    logging.info(f"Creating the graph took {time.time() - start_time} seconds")
    logging.info(f"Graph has {tot_nodes} nodes and {tot_edges} edges")

    start_time = time.time()
    from tools.dimacs_to_metis import convert

    # TODO: we could also directly write the graph to metis, but shouldn't change much
    metis = convert(dimacs_graph)
    open("graph-sorted.graph", "w").write(metis)
    logging.debug("Converted graph to metis format")
    # TODO: test different parameters for redumis (see them with --help)
    return_code = os.system(
        "tools/redumis graph-sorted.graph --output=graph-sorted.graph.out --disable_checks"
        + " > /dev/null" * (not DEBUG)
    )
    if return_code != 0:
        raise Exception("Redumis failed. Run with --log_level=DEBUG for more info.")
    if csv_logger:
        csv_logger.log("graph_solving_time", time.time() - start_time)
    out_file = open("graph-sorted.graph.out", "r")
    invisible_set = set(
        [points[i] for i in range(len(points)) if out_file.readline().strip() == "1"]
    )
    if csv_logger:
        csv_logger.log("num_invisible_points", len(invisible_set))
    out_file.close()

    if not DEBUG:
        os.remove("graph-sorted.graph")
        os.remove("graph-sorted.graph.out")

    logging.info(
        f"Finding the maximum independent set took {time.time() - start_time} seconds"
    )
    return invisible_set


def find_invisible_points(
    p: Polygon,
    sampling_method: str = "default",
    num_samples: int = 100,
    csv_logger: CsvLogger = None,
) -> set[Point]:
    """
    Finds a set of points that are mutually invisible with each other within the polygon.
    :param p: the polygon
    :param sampling_method: the method used to sample initial points from the polygon
    :param num_samples: the number of initial points to sample. Only used for random sampling

    :return: a set of points that are invisible from the polygon
    """
    if sampling_method == "default":
        sampling_method = "midpoint"
    start_time = time.time()
    initial_points = []
    if sampling_method == "alternating":
        initial_points = sample_alternating_holes_points(p)
    elif sampling_method == "random":
        initial_points = sample_random_points(p, num_samples)
    elif sampling_method == "midpoint":
        initial_points = sample_mid_points(p)

    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    if csv_logger:
        csv_logger.log("sampling_method", sampling_method)
        csv_logger.log("sampling_time", time.time() - start_time)
    logging.info(
        f"Sampling {len(initial_points)} points took {time.time() - start_time} seconds (sampling method: {sampling_method})"
    )
    if DEBUG:
        plot_polygon_and_points(p, initial_points, colour_style="b.")
    invisible_points = find_invisible_points_within_set(
        p, initial_points, can_see=can_see, csv_logger=csv_logger
    )
    return invisible_points


def plot_polygon_and_points(p: Polygon, points: set, colour_style: str = "r."):
    """
    Plots the polygon and the invisible points
    """
    # plot the polygon
    plt.plot(*p.exterior.xy, "k-")

    hole_style = "k-" if not DEBUG else "k--"
    for hole in p.interiors:
        plt.plot(*hole.xy, hole_style)

    # plot the invisible points
    for point in points:
        plt.plot(*point.xy, colour_style)


def points_to_json(
    polygon_name: str, points: set, sampling_method: str, total_runtime: float
) -> dict:
    """
    Converts a set of points to a json object
    """
    return {
        "name": polygon_name,
        "points": [{"x": point.x, "y": point.y} for point in points],
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "machine": os.uname()[1],
        "version": VERSION,
        "sampling_method": sampling_method,
        "total_runtime": total_runtime,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sampling_method",
        type=str,
        help="The method used to sample initial points from the polygon",
        choices=["alternating", "random", "midpoint"],
    )
    parser.add_argument(
        "filenames",
        type=str,
        nargs="+",
        help="The name of the json files containing the polygons",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="The number of initial points to sample. Only used for random sampling",
        default=100,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        help="The level of logging",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="The name of the folder where to save the output files",
        default=False,
    )
    args = parser.parse_args()
    sampling_method = args.sampling_method
    filenames = args.filenames
    num_samples = args.num_samples
    csv_logger = None
    if args.save:
        if os.path.exists(args.save):
            answer = input(
                f"The folder {args.save} already exists. Do you want to overwrite it? (y/n) "
            )
            if answer != "y":
                print("Exiting")
                exit()
        os.makedirs(args.save, exist_ok=True)
        csv_logger = CsvLogger(args.save)

    logging.basicConfig(level=args.log_level)
    global DEBUG
    DEBUG = logging.getLogger().isEnabledFor(logging.DEBUG)
    try:
        for f in filenames:
            if csv_logger:
                csv_logger.set_instance(f)

            start_time = time.time()
            p = load_polygon(f, csv_logger)
            invisible_set = find_invisible_points(
                p,
                sampling_method=sampling_method,
                num_samples=num_samples,
                csv_logger=csv_logger,
            )
            logging.info(f"Found {len(invisible_set)} invisible points in {f}")
            if args.save:
                csv_logger.log("total_time", time.time() - start_time)
                file_path = f"{args.save}/out_{f.split('/')[-1]}"
                write_file = True
                if os.path.exists(file_path):
                    with open(file_path, "r") as file:
                        data = json.load(file)
                        if len(data["points"]) < len(invisible_set):
                            logging.info(
                                f"Overwriting {file_path} because it contains less points than the current set"
                            )
                        else:
                            write_file = False
                if write_file:
                    json.dump(
                        points_to_json(
                            f,
                            invisible_set,
                            sampling_method,
                            time.time() - start_time,
                        ),
                        open(file_path, "w"),
                    )
            if len(filenames) == 1:
                plot_polygon_and_points(p, invisible_set)
    finally:
        if csv_logger:
            csv_logger.save()
    plt.show()


if __name__ == "__main__":
    main()

# --------------------- TESTS ---------------------


@pytest.mark.parametrize("sampling_method", ["default", "alternating", "random"])
def test_find_invisible_points(sampling_method):
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    invisible_set = find_invisible_points(
        p, sampling_method=sampling_method, num_samples=50
    )
    assert len(invisible_set) == 1


def test_can_see():
    p = Polygon(
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        holes=[[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]],
    )
    assert can_see(p, Point(0.1, 0.1), Point(0.9, 0.1))
    assert not can_see(p, Point(0.1, 0.1), Point(0.9, 0.9))
    assert can_see(p, Point(0.2, 0.2), Point(0.8, 0.2))
    assert not can_see(p, Point(0.2, 0.2), Point(0.8, 0.8))
    assert can_see(p, Point(0, 0), Point(0.2, 0.2))
    assert can_see(p, Point(0, 0), Point(0, 1))
    assert not can_see(p, Point(0, 0), Point(0.8, 0.8))
    assert not can_see(p, Point(0, 0), Point(1, 1))
