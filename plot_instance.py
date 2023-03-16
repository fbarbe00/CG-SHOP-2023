import json
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from cgshop2023_pyutils import InstanceDatabase, verify

# matplotlib.use('TkAgg')


def plot_solution(
    instance_name: str,
    exterior: np.ndarray,
    holes: list[np.ndarray],
    polygons: list[np.ndarray],
    save_to_path: str = "",
):
    cmap = plt.get_cmap("Set1")

    fig, ax = plt.subplots(figsize=(20, 15))

    exterior = Polygon(
        exterior, facecolor="none", closed=True, edgecolor=cmap(8), linewidth=1.0
    )
    ax.add_patch(exterior)

    for h in holes:
        h = Polygon(
            h,
            facecolor=cmap(1),
            alpha=0.25,
            closed=True,
            edgecolor=cmap(8),
            linewidth=1.0,
        )
        ax.add_patch(h)

    for p in polygons:
        p = Polygon(
            p,
            facecolor=cmap(2),
            alpha=0.2,
            closed=True,
            edgecolor=cmap(6),
            linewidth=0.5,
        )
        ax.add_patch(p)

    ax.autoscale()

    plt.title(f"{instance_name}\n #hulls {len(polygons)}", fontsize=30)

    plt.gca().axis("off")
    plt.tight_layout()

    if save_to_path:
        plt.savefig(save_to_path, dpi=200)
    else:
        plt.show()


def main():
    instance_name = "srpg_iso_aligned_mc0094745"

    if len(sys.argv) == 2:
        instance_name = sys.argv[1]

    plots_path = "outputs/plots/"
    os.makedirs(plots_path, exist_ok=True)

    idb = InstanceDatabase("./inputs/competition_instances")
    instance = idb[instance_name]

    solution_path = glob.glob(f"outputs/coverage/*{instance_name}*.json")
    assert len(solution_path) == 1
    solution_path = solution_path[0]

    solution_data = json.load(open(solution_path, "r"))

    polygons = []
    for pl in solution_data["polygons"]:
        polygons.append(np.array([[p["x"], p["y"]] for p in pl]))

    holes = []
    for inter in instance["holes"]:
        holes.append(np.array([[v["x"], v["y"]] for v in inter]))

    exterior = np.array([[v["x"], v["y"]] for v in instance["outer_boundary"]])

    plot_solution(
        instance_name,
        exterior,
        holes,
        polygons,
        os.path.join(plots_path, f"{instance_name}.png"),
    )


if __name__ == "__main__":
    main()
