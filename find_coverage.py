import json
import time
import os
import logging
from typing import Union
import random
import argparse
from fractions import Fraction

from tqdm import tqdm
from datetime import datetime
import pandas as pd
from scipy.spatial import distance, ConvexHull
import triangle as tr
import numpy as np
import networkx as nx
from shapely import geometry, Polygon
from joblib import Memory

from cgshop2023_pyutils import InstanceDatabase, verify

SEED = 1
VERSION = "0.1.9"
SOLUTION_FOLDER = "outputs/coverage"
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s | %(name)s | %(levelname)s] %(message)s",
)
os.makedirs(SOLUTION_FOLDER, exist_ok=True)
memory = Memory("./tcache", verbose=0)


class CsvLogger:
    def __init__(self, output_folder: str):
        self.df = pd.DataFrame(
            columns=[
                "version",
                "instance",
                "machine",
                "search_method",
                "triangulation_opts",
                "initial_solution_id",
                "solution_size",
                "overlapping_polygons",
                "cover_time",
                "solution_time",
                "overlapping_polygons_time",
                "total_time",
                "seed",
                "enable_scaling",
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
            os.path.join(
                self.output_folder,
                "alllog" + datetime.now().strftime("%Y%m%d%H%M%S") + ".csv",
            ),
            index=False,
        )

    def remove_instance(self):
        self.df.drop(self.current_instance, inplace=True)
        self.current_instance = None


def polygon_area(polygon: np.ndarray) -> float:
    return (
        1
        / 2
        * (
            polygon[:, 0] @ np.roll(polygon[:, 1], 1)
            - polygon[:, 1] @ np.roll(polygon[:, 0], 1)
        )
    )


def compute_centroid(polygon: np.ndarray) -> np.ndarray:
    a = polygon_area(polygon)
    k = polygon[:, 0] * np.roll(polygon[:, 1], 1) - polygon[:, 1] * np.roll(
        polygon[:, 0], 1
    )

    c_x = 1 / (6 * a) * (polygon[:, 0] + np.roll(polygon[:, 0], 1)) @ k
    c_y = 1 / (6 * a) * (polygon[:, 1] + np.roll(polygon[:, 1], 1)) @ k

    return np.array([c_x, c_y])


def create_vertices_segments(
    polygons: list[np.ndarray],
) -> list[np.ndarray, np.ndarray]:
    shift = 0
    segments = []
    for p in polygons:
        n = np.arange(len(p))
        segments.append(np.vstack([n, n + 1]).T % len(n) + shift)
        shift += len(p)

    return [np.vstack(polygons), np.vstack(segments)]


def make_vertices_graph(triangulation: dict) -> nx.Graph:
    graph = nx.Graph()
    triangles = triangulation["triangles"]

    for n in range(0, len(triangles)):
        vertices = triangulation["vertices"][triangles[n]]
        graph.add_node(n, vertices=vertices)

    # sort triangles indexes
    for i in range(len(triangles)):
        triangles[i] = np.sort(triangles[i])

    vertices_map = {idx: [] for idx in range(len(triangulation["vertices"]))}
    for i in range(len(triangles)):
        t = np.array(triangles[i])
        for v in t:
            vertices_map[v].append([i, t])

    for i in tqdm(range(len(triangles))):
        for vertex_idx in triangles[i]:
            for tr_idx, tr_adj in vertices_map[vertex_idx]:
                if np.all(triangles[i] == tr_adj):
                    continue

                graph.add_edge(i, tr_idx)

    return graph


@memory.cache
def make_triangles_graph(triangulation: dict) -> nx.Graph:
    graph = nx.Graph()
    triangles = triangulation["triangles"]

    for n in range(0, len(triangles)):
        vertices = triangulation["vertices"][triangles[n]]
        graph.add_node(n, vertices=vertices)

    # sort triangles indexes
    for i in range(len(triangles)):
        triangles[i] = np.sort(triangles[i])

    triangles_map = {idx: [] for idx in range(len(triangles))}
    for i in range(len(triangles)):
        t = np.array(triangles[i])
        for v in t:
            triangles_map[v].append([i, t])

    triangles_connections = [0] * len(triangles)
    for i in tqdm(range(len(triangles))):
        if triangles_connections[i] == 3:
            # we cannot have more than 3 adjacencies
            continue

        for vertex_idx in triangles[i]:
            for tr_idx, tr_adj in triangles_map[vertex_idx]:
                if np.all(triangles[i] == tr_adj):
                    continue

                if triangles_connections[tr_idx] == 3:
                    # we cannot have more than 3 adjacencies
                    continue

                if triangles_connections[i] == 3:
                    # we cannot have more than 3 adjacencies
                    break

                if np.sum(np.isin(triangles[i], tr_adj, assume_unique=True)) >= 2:
                    # the triangle j is adjacent to the triangle i
                    triangles_connections[i] += 1
                    triangles_connections[tr_idx] += 1
                    graph.add_edge(i, tr_idx)

    return graph


def triangle_ids_to_coords(
    triangulation: dict, nodes: Union[int, np.ndarray, list]
) -> np.ndarray:
    vertices_ids = triangulation["triangles"][nodes].flatten()
    vertices = triangulation["vertices"][vertices_ids]
    return vertices


def hull_valid(
    triangulation: dict,
    triangle_ids: list,
    exterior: geometry.Polygon,
    holes: list[geometry.Polygon],
) -> [bool, float]:
    vertices = triangle_ids_to_coords(triangulation, triangle_ids)
    hull = ConvexHull(vertices)
    hull_vertices = hull.points[hull.vertices]
    shapely_hull = geometry.Polygon(hull_vertices)

    exterior_intersection = (
        shapely_hull.area - (shapely_hull.intersection(exterior)).area
    )
    if exterior_intersection > 1e-6:
        return False, float("inf")

    for hl in holes:
        if shapely_hull.intersects(hl):
            inters = shapely_hull.intersection(hl)
            if inters.area > 1e-6:
                return False, inters.area

    return True, 0.0


def search_cover(
    triangulation: dict,
    exterior: geometry.Polygon,
    holes: list[geometry.Polygon],
    triangles_graph: nx.Graph,
    source_node: int,
    search_method: str = "bfs",
) -> [ConvexHull, list]:

    if search_method == "bfs":
        tree = nx.bfs_tree(triangles_graph, source=source_node, depth_limit=None)
    elif search_method == "dfs":
        tree = nx.dfs_tree(triangles_graph, source=source_node, depth_limit=None)
    else:
        raise ValueError("Unknown search method: {}".format(search_method))

    cover = []
    nodes_to_explore = [source_node]

    while len(nodes_to_explore):
        cover.append(nodes_to_explore.pop())
        valid, inter_area = hull_valid(triangulation, cover, exterior, holes)

        if valid:
            nodes_to_explore += list(tree.neighbors(cover[-1]))
        else:
            nodes_to_explore += list(tree.neighbors(cover[-1]))
            cover.pop()

    return ConvexHull(triangle_ids_to_coords(triangulation, cover)), cover


def triangulate(vertices, segments, holes, triangulation_opts):
    if len(holes):
        return tr.triangulate(
            {"vertices": vertices, "segments": segments, "holes": holes},
            triangulation_opts,
        )
    else:
        return tr.triangulate(
            {"vertices": vertices, "segments": segments}, triangulation_opts
        )


def compute(
    exterior: np.ndarray,
    holes: list[np.ndarray],
    starting_points: list,
    triangulation_opts: str = "pa",
    search_method: str = "bfs",
) -> ConvexHull:

    logger.info("Initial preparations")

    points_from_holes = []
    for h in holes:

        # take any convex polygon inside a hole
        ts = tr.triangulate({"vertices": h})
        ts = ts["vertices"][ts["triangles"]]

        pt = None
        for t in ts:
            # a naive way to check that a centroid is inside the polygon
            plgn = geometry.Polygon(geometry.LineString(h))
            pt = geometry.Point(compute_centroid(t))
            if plgn.contains(pt):
                break

        points_from_holes.append(np.hstack(pt.coords.xy))

    logger.info("Triangulation")

    points_from_holes = np.array(points_from_holes)
    v, s = create_vertices_segments([exterior] + holes)

    triangulation = triangulate(v, s, points_from_holes, triangulation_opts)

    logger.info("Making a graph")

    triangles_graph = make_vertices_graph(triangulation)
    # triangles_graph = make_triangles_graph(triangulation)

    logger.info("Working on cover")

    cover = []
    triangles_ids = set(range(len(triangulation["triangles"])))
    holes = [geometry.Polygon(h) for h in holes]
    exterior = geometry.Polygon(exterior)

    starting_traingles = set()
    for p in starting_points:
        p = np.array([p])
        best_triangle = None
        min_dist = float("inf")
        for ti in triangles_ids:
            vs = triangulation["vertices"][triangulation["triangles"][ti]]
            d = np.min(distance.cdist(p, vs, "sqeuclidean"))
            if d < min_dist:
                best_triangle = ti
                min_dist = d

        starting_traingles.add(best_triangle)
    np.random.seed(SEED)
    random.seed(SEED)  # FIXME: remove

    pbar = tqdm(total=len(triangles_ids))
    while len(triangles_ids):
        if starting_traingles:
            source_node = starting_traingles.pop()
        else:
            source_node = random.sample(list(triangles_ids), 1)[0]

        hull, hull_triangle_ids = search_cover(
            triangulation, exterior, holes, triangles_graph, source_node, search_method
        )
        s1 = len(triangles_ids)

        triangles_ids.difference_update(hull_triangle_ids)
        starting_traingles.difference_update(hull_triangle_ids)

        pbar.update(s1 - len(triangles_ids))
        cover.append(hull)

    pbar.close()
    logger.info(f"The solution size: {len(cover)}")

    return cover


def generate_solution(
    instance: dict,
    starting_points: list,
    triangulation_opts: str,
    search_method: str = "bfs",
    enable_scaling: bool = False,
    csv_logger: CsvLogger = None,
):

    exterior = np.array([[v["x"], v["y"]] for v in instance["outer_boundary"]])
    holes = []
    for inter in instance["holes"]:
        holes.append(np.array([[v["x"], v["y"]] for v in inter]))

    scale_factor = np.float64(1.0)
    if enable_scaling:
        scale_factor = np.max(exterior).astype("float64") / 10

    exterior = exterior.astype("float64") / scale_factor.astype("float64")
    holes = [h.astype("float64") / scale_factor.astype("float64") for h in holes]

    # triangulation_opts = qpa100, pa500 (see https://rufat.be/triangle/API.html)
    start_time = time.time()
    cover = compute(exterior, holes, starting_points, triangulation_opts, search_method)
    if csv_logger:
        csv_logger.log("cover_time", time.time() - start_time)

    solution = {
        "type": "CGSHOP2023_Solution",
        "instance": instance["name"],
        "polygons": [],
    }
    start_time = time.time()

    for c in cover:
        points = (c.points[c.vertices] * scale_factor).tolist()
        polygon = []
        for i, p in enumerate(points):
            x = Fraction(p[0])
            y = Fraction(p[1])
            pd = {}

            if x.denominator == 1:
                pd["x"] = int(x)
            else:
                pd["x"] = {"num": x.numerator, "den": x.denominator}

            if y.denominator == 1:
                pd["y"] = int(y)
            else:
                pd["y"] = {"num": y.numerator, "den": y.denominator}

            polygon.append(pd)

        solution["polygons"].append(polygon)

    if csv_logger:
        csv_logger.log("solution_time", time.time() - start_time)

    return solution


def load_starting_points(starting_points_path) -> list:
    data = json.load(open(starting_points_path, "r"))
    if "guards" in data:
        # From Kasper
        return data["guards"]
    else:
        # From Fabio
        ps = []
        for p in data["points"]:
            ps.append([p["x"], p["y"]])

        return ps


def find_overlapping_polygons(polygons: list) -> set:
    """
    Find all the polygons that are fully contained by other polygons.
    """
    contained_polygons = set()
    polygons_shapely = []
    for poly in tqdm(polygons):
        for p in poly:
            if type(p["x"]) == dict and "num" in p["x"]:
                p["x"] = p["x"]["num"] / p["x"]["den"]
            if type(p["y"]) == dict and "num" in p["y"]:
                p["y"] = p["y"]["num"] / p["y"]["den"]
        polygons_shapely.append(Polygon([(p["x"], p["y"]) for p in poly]))

    for i in tqdm(range(len(polygons))):
        p1 = polygons_shapely[i]
        if p1.is_empty:
            contained_polygons.add(i)
            continue
        polygons_shapely_copy = polygons_shapely.copy()
        for j in range(i + 1, len(polygons)):
            p2 = polygons_shapely[j]
            polygons_shapely[j] = p2.difference(polygons_shapely[i])
            p1 = p1.difference(p2)
            if p1.is_empty:
                contained_polygons.add(i)
                polygons_shapely = polygons_shapely_copy
                break

    return contained_polygons


def save_solution(solution: dict, force_overwrite=False):
    solution_path = os.path.join(
        SOLUTION_FOLDER,
        f'{solution["instance"]}_best_solution.json',
    )
    if not force_overwrite and os.path.exists(solution_path):
        old_solution = json.load(open(solution_path, "r"))
        if len(old_solution["polygons"]) <= len(solution["polygons"]):
            return
    json.dump(solution, open(solution_path, "w"))


def main():
    COMP_FOLDER = "inputs/competition_instances"
    csv_logger = CsvLogger(SOLUTION_FOLDER)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "instance_names",
        type=str,
        nargs="+",
        help=f"The names of the polygon instances (in the {COMP_FOLDER} folder)",
    )
    parser.add_argument(
        "--triangulation_opts",
        type=str,
        default="pq0",
        help="The triangulation options. See https://rufat.be/triangle/API.html",
    )
    parser.add_argument(
        "--search_method",
        type=str,
        choices=["bfs", "dfs"],
        default="bfs",
        help="The search method for merging triangles",
    )
    parser.add_argument(
        "--enable_scaling",
        type=bool,
        default=False,
        help="Enable scaling",
    )
    parser.add_argument(
        "--starting_points",
        type=str,
        default=None,
        help="Folder containing the starting points",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for the random number generator",
    )
    args = parser.parse_args()
    SEED = args.seed

    try:
        for instance_name in args.instance_names:
            logger.info(f"Generating solution for {instance_name}")
            starting_points = []
            idb = InstanceDatabase("inputs/competition_instances")
            instance = idb[instance_name]
            csv_logger.set_instance(instance_name)
            csv_logger.log("search_method", args.search_method)
            csv_logger.log("seed", args.seed)
            csv_logger.log("triangulation_opts", args.triangulation_opts)
            csv_logger.log("initial_solution_id", args.starting_points)
            csv_logger.log("enable_scaling", args.enable_scaling)
            start_time = time.time()
            if args.starting_points:
                starting_points_path = os.path.join(
                    args.starting_points, f"out_{instance_name}.instance.json"
                )
                starting_points = load_starting_points(starting_points_path)
            solution = generate_solution(
                instance,
                starting_points,
                triangulation_opts=args.triangulation_opts,
                search_method=args.search_method,
                enable_scaling=args.enable_scaling,
                csv_logger=csv_logger,
            )
            csv_logger.log("total_time", time.time() - start_time)
            err_msg = verify(instance, solution)
            if err_msg:
                logger.error("SOLUTION INVALID:", err_msg)
                csv_logger.log("solution_size", err_msg)
            else:
                csv_logger.log("solution_size", len(solution["polygons"]))
                save_solution(solution)
                start_time = time.time()
                overlapping_polygons = find_overlapping_polygons(solution["polygons"])
                solution["polygons"] = [
                    solution["polygons"][i]
                    for i in range(len(solution["polygons"]))
                    if i not in overlapping_polygons
                ]
                err_msg = verify(instance, solution)
                if err_msg:
                    logger.error("SOLUTION INVALID:", err_msg)
                    csv_logger.log("overlapping_polygons", err_msg)
                else:
                    csv_logger.log("overlapping_polygons", len(overlapping_polygons))
                    save_solution(solution)
                    logger.info(
                        f"Removed {len(overlapping_polygons)} overlapping polygons"
                    )
                csv_logger.log("overlapping_polygons_time", time.time() - start_time)
    finally:
        csv_logger.save()


if __name__ == "__main__":
    main()
