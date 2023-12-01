import fire
import pandas as pd
import h5py
import os
from epam import evaluation, models, esm_precompute
from epam.utils import generate_file_checksum


def aaprob(model_name, model_args, in_path, out_path, hdf5_path=None):
    """
    Generate amino acid probability matrices using the specified model.

    Args:
        model_name (str): Name of the model class to use.
        model_args (str): JSON string of arguments to pass to the model constructor.
        in_path (str): Path to the input file.
        out_path (str): Path where the output file will be written.
        hdf5_path (str, optional): Path to the HDF5 file containing precomputed ESM1v selection factors.

    Examples:
        epam aaprob SHMple '{"weights_directory":"data/shmple_weights/my_shmoof"}' in_path out_path
    """
    ModelClass = getattr(models, model_name)
    model = ModelClass(**model_args)
    if model_name in ("CachedESM1v", "SHMpleESM"):
        if hdf5_path is None:
            raise ValueError(
                f"Model {model_name} requires an HDF5 file containing precomputed ESM1v selection factors."
            )
        print("Preloading ESM1v data...")
        model.preload_esm_data(hdf5_path)
    model.write_aaprobs(in_path, out_path)


def evaluate(aaprob_paths_str, model_performance_path):
    aaprob_paths = aaprob_paths_str.split(",")
    evaluation.evaluate(aaprob_paths, model_performance_path)


def concatenate_csvs(
    input_csvs_str: str,
    output_csv: str,
    is_tsv: bool = False,
    record_path: bool = False,
):
    """
    This function concatenates multiple CSV or TSV files into one CSV file.
    :param input_csvs: A string of paths to the input CSV or TSV files separated by commas.
    :param output_csv: Path to the output CSV file.
    :param is_tsv: A boolean flag that determines whether the input files are TSV.
    :param record_path: A boolean flag that adds a column recording the path of the input_csv.
    """
    input_csvs = input_csvs_str.split(",")
    dfs = []

    for csv in input_csvs:
        df = pd.read_csv(csv, delimiter="\t" if is_tsv else ",")
        if record_path:
            df["input_file_path"] = csv
        dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)

    result_df.to_csv(output_csv, index=False)


def concatenate_hdf5s(input_files, output_file):
    """
    This function concatenates multiple HDF5 files into a single HDF5 file. Used to combine aaprobs HDF5 files across batches of PCPs.

    Parameters:
    :param input_files (str): A string of paths to input HDF5 files.
    :param output_file (str): Path to the output merged HDF5 file.
    """
    input_hdf5s = input_files.split(",")

    with h5py.File(output_file, 'w') as merged_file:
        checksums = []
        model_names = set()
        pcp_filenames = []
        for input_file in input_hdf5s:
            with h5py.File(input_file, 'r') as input_hdf5:
                checksums.append(input_hdf5.attrs["checksum"])
                model_names.add(input_hdf5.attrs["model_name"])
                pcp_filenames.append(input_hdf5.attrs["pcp_filename"])

                for dataset_name, dataset in input_hdf5.items():
                    merged_file.copy(dataset, dataset_name)

        if len(model_names) == 1:
            merged_model_name = model_names.pop()
        else:
            raise ValueError("Model names do not match across input files.")

        common_prefix = os.path.commonprefix(pcp_filenames)
        full_pcp_file_path = common_prefix.replace("pcp_batched_inputs", "pcp_inputs").rsplit("_", 1)[0] + ".csv"
        full_pcp_checksum = generate_file_checksum(full_pcp_file_path)

        merged_file.attrs["checksum"] = full_pcp_checksum
        merged_file.attrs["model_name"] = merged_model_name
        merged_file.attrs["pcp_filename"] = full_pcp_file_path


def esm_bulk_precompute(csv_path, output_hdf5_path):
    """
    This subcommand precomputes ESM1v selection factors for a set of PCPs in bulk, and then saves those values in an HDF5 file for later use in SHMple-ESM.

    :param csv_path: Path to a CSV file containing PCP data.
    """
    esm_precompute.precompute_and_save(csv_path, output_hdf5_path)


def main():
    fire.Fire()
