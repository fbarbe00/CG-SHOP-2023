import os
import csv

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# matplotlib.use('TkAgg')


def plot_solution(data: list, save_to_path: str = ""):
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(10, 7))

    data = sorted(data, key=lambda x: x["solution_size"])

    names = [x["instance"] for x in data]
    solutions = [x["solution_size"] for x in data]
    lowerbounds = [x["lowerbound"] for x in data]
    upperbounds1 = [x["upperbound1"] for x in data]
    upperbounds2 = [x["upperbound2"] for x in data]
    xs = list(range(len(solutions)))

    ax.plot(xs, solutions, ".", color=cmap(0.0))
    ax.fill_between(xs, lowerbounds, upperbounds1, color=cmap(0.5), alpha=0.2)
    ax.fill_between(xs, lowerbounds, upperbounds2, color=cmap(0.1), alpha=0.2)

    ax.set_xticklabels(names, rotation=45)
    ax.xaxis.set_major_locator(MultipleLocator(7))
    ax.tick_params(axis="x", labelsize=7)

    ax.set_yscale("log")

    ax.legend(["Solution size", "Theoretical Range 1", "Theoretical Range 2"])

    ax.autoscale()
    plt.title(f"Results", fontsize=30)
    plt.tight_layout()

    plt.savefig(save_to_path, dpi=200)
    # plt.show()


def main():
    results_path = "results.csv"
    plots_path = "outputs/plots/"
    os.makedirs(plots_path, exist_ok=True)

    data = csv.DictReader(open(results_path, "r"), delimiter=";")
    data = list(data)

    for d in data:
        for k, v in d.items():
            if k != "instance":
                d[k] = int(v)

    plot_solution(data, os.path.join(plots_path, "results.png"))


if __name__ == "__main__":
    main()
