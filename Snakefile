import json
import subprocess

model_info = [
    ("AbLang_heavy", "AbLang", {"chain": "heavy"}),
    ("SHMple_default", "SHMple", {"weights_directory": "data/shmple_weights/my_shmoof"})
    ("SHMple_productive", "SHMple", {"weights_directory": "data/shmple_weights/prod_shmple"})
]

model_name_to_spec = {
    model_name: [model_class, json.dumps({**model_params, "model_name": model_name})]
    for model_name, model_class, model_params in model_info
}

pcp_inputs = glob_wildcards("pcp_inputs/{name}.csv").name


rule all:
    input:
        "output/combined_performance.csv",


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
