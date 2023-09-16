import os

from matplotlib.backends.backend_pdf import PdfPages
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF


def summarize_pcps(input_path, output_path):
    df = pd.read_csv(input_path)

    df["mutation_count"] = 0
    df["sequence_length"] = 0
    df["mutation_frac"] = 0.0

    for index, row in df.iterrows():
        parent = row["parent"]
        child = row["child"]

        if len(parent) != len(child):
            raise ValueError("Parent and child sequences must be of the same length.")

        sequence_length = sum(1 for p, c in zip(parent, child) if p != "N" and c != "N")
        mutation_count = sum(
            1 for p, c in zip(parent, child) if p != c and p != "N" and c != "N"
        )

        df.at[index, "mutation_count"] = mutation_count
        df.at[index, "sequence_length"] = sequence_length
        df.at[index, "mutation_frac"] = (
            mutation_count / sequence_length if sequence_length > 0 else 0
        )

    with PdfPages(output_path) as pdf:
        # Plot the distribution of sequence lengths
        plt.figure(figsize=(10, 6))
        plt.hist(df["sequence_length"], bins=30, color="skyblue", edgecolor="black")
        plt.title(
            f'Sequence Length Distribution\n{os.path.basename(input_path).replace(".csv", "")}'
        )
        plt.xlabel("Sequence Length")
        plt.ylabel("Frequency")
        pdf.savefig()
        plt.close()

        # Plot the distribution of mutation fractions
        plt.figure(figsize=(10, 6))
        plt.hist(df["mutation_frac"], bins=30, color="lightcoral", edgecolor="black")
        plt.title(
            f'Mutation Fraction Distribution\n{os.path.basename(input_path).replace(".csv", "")}'
        )
        plt.xlabel("Mutation Fraction")
        plt.ylabel("Frequency")
        pdf.savefig()
        plt.close()

        # Plot the ECDF of mutation fractions
        plt.figure(figsize=(10, 6))
        ecdf = ECDF(df["mutation_frac"])
        plt.step(ecdf.x, ecdf.y, color="mediumpurple")

        # Add horizontal lines at 0.1 intervals
        for y in np.arange(0.1, 1.1, 0.1):  # 1.1 is used to include 1.0 as well
            plt.axhline(y, color="grey", linestyle="dashed", linewidth=0.5)

        plt.title(
            f'Empirical CDF of Mutation Fraction\n{os.path.basename(input_path).replace(".csv", "")}'
        )
        plt.xlabel("Mutation Fraction")
        plt.ylabel("ECDF")
        pdf.savefig()
        plt.close()


if __name__ == "__main__":
    fire.Fire(summarize_pcps)
