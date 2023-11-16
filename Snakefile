import epam.models
import json
import subprocess
from importlib import resources


with resources.path("epam", "__init__.py") as p:
    DATA_DIR = str(p.parent.parent) + "/data/"


local_FULLY_SPECIFIED_MODELS = [
    ("AbLang_heavy", "AbLang", {"chain": "heavy"}),
    (
        "SHMple_default",
        "SHMple",
        {"weights_directory": DATA_DIR + "shmple_weights/my_shmoof"},
    ),
    (
        "SHMple_productive",
        "SHMple",
        {"weights_directory": DATA_DIR + "shmple_weights/prod_shmple"},
    ),
    ("ESM1v_default", "CachedESM1v", {"hdf5_path": pcp_hdf5_path}),
    (
        "SHMple_ESM1v",
        "SHMpleESM",
        {
            "hdf5_path": pcp_hdf5_path,
            "weights_directory": DATA_DIR + "shmple_weights/my_shmoof",
        },
    ),
]

model_name_to_spec = {
    model_name: [model_class, json.dumps({**model_params, "model_name": model_name})]
    # for model_name, model_class, model_params in epam.models.FULLY_SPECIFIED_MODELS
    for model_name, model_class, model_params in local_FULLY_SPECIFIED_MODELS
}

pcp_inputs = glob_wildcards("pcp_inputs/{name}.csv").name


rule all:
    input:
        "output/combined_performance.csv",


rule precompute_esm:
    input:
        in_csv="pcp_inputs/{pcp_input}.csv",
    output:
        out_hdf5="pcp_inputs/{pcp_input}.hdf5",
    shell:
        """
        epam esm_bulk_precompute {input.in_csv}
        """


rule run_model:
    input:
        in_csv="pcp_inputs/{pcp_input}.csv",
    output:
        aaprob="output/{pcp_input}/{model_name}/aaprob.hdf5",
        performance="output/{pcp_input}/{model_name}/performance.csv",
    params:
        model_class=lambda wildcards: model_name_to_spec[wildcards.model_name][0],
        model_params=lambda wildcards: model_name_to_spec[wildcards.model_name][1],
    benchmark:
        "output/{pcp_input}/{model_name}/timing.tsv"
    shell:
        """
        mkdir -p output/{wildcards.pcp_input}/{wildcards.model_name}
        epam aaprob {params.model_class} '{params.model_params}' {input.in_csv} {output.aaprob}
        epam evaluate {output.aaprob} {output.performance}
        """


rule combine_performance_files:
    input:
        expand(
            "output/{pcp_input}/{model_name}/performance.csv",
            pcp_input=pcp_inputs,
            model_name=model_name_to_spec.keys(),
        ),
    output:
        "output/combined_performance.csv",
        "output/combined_timing.csv",
    run:
        input_files = ",".join(input)
        input_timing_files = ",".join(
            f"output/{pcp_input}/{model_name}/timing.tsv"
            for pcp_input in pcp_inputs
            for model_name in model_name_to_spec.keys()
        )
        output_file = output[0]
        output_timing_file = output[1]
        subprocess.run(
            f"epam concatenate_csvs {input_files} {output_file}", shell=True, check=True
        )
        subprocess.run(
            f"epam concatenate_csvs {input_timing_files} {output_timing_file} --is_tsv --record_path",
            shell=True,
            check=True,
        )
