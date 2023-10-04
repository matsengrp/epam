import fire
import pandas as pd
from epam import evaluation, models


def aaprob(model_name, model_args, in_path, out_path):
    """
    Generate amino acid probability matrices using the specified model.

    Args:
        model_name (str): Name of the model class to use.
        model_args (str): JSON string of arguments to pass to the model constructor.
        in_path (str): Path to the input file.
        out_path (str): Path where the output file will be written.

    Examples:
        epam aaprob SHMple '{"weights_directory":"data/shmple_weights/my_shmoof"}' in_path out_path
    """
    ModelClass = getattr(models, model_name)
    model = ModelClass(**model_args)
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


def main():
    fire.Fire()
