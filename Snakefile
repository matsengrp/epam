import epam.models
import json
import subprocess

# add rule to split pcp files into subsets of reasonable size, merge before evaluation
# aaprob and evaluate will then need to be separated for each model

model_name_to_spec = {
    model_name: [model_class, json.dumps({**model_params, "model_name": model_name})]
    for model_name, model_class, model_params in epam.models.FULLY_SPECIFIED_MODELS 
}

set1_models = ("AbLang_heavy", "ESM1v_default")
set2_models = ("SHMple_default", "SHMple_productive")
set3_models = "SHMple_ESM1v"

set1_model_name_to_spec = {
    key: model_name_to_spec[key] for key in set1_models
}

set2_model_name_to_spec = {
    key: model_name_to_spec[key] for key in set2_models
}

set3_model_name_to_spec = {
    set3_models: model_name_to_spec[set3_models]
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
        epam esm_bulk_precompute {input.in_csv} {output.out_hdf5}
        """


rule run_model_set1:
    input:
        in_csv="pcp_inputs/{pcp_input}.csv",
        hdf5_path="pcp_inputs/{pcp_input}.hdf5",
    output:
        touch("{pcp_input}_{model_name}.done"),
        aaprob="output/{pcp_input}/set1/{model_name}/aaprob.hdf5",
        performance="output/{pcp_input}/set1/{model_name}/performance.csv",
    params:
        model_class=lambda wildcards: set1_model_name_to_spec[wildcards.model_name][0],
        model_params=lambda wildcards: set1_model_name_to_spec[wildcards.model_name][1],
    benchmark:
        "output/{pcp_input}/set1/{model_name}/timing.tsv"
    shell:
        """
        mkdir -p output/{wildcards.pcp_input}/set1/{wildcards.model_name}
        epam aaprob {params.model_class} '{params.model_params}' {input.in_csv} {output.aaprob} {input.hdf5_path}
        epam evaluate {output.aaprob} {output.performance}
        """


rule run_model_set2:
    input:
        expand("{{pcp_input}}_{model}.done", model = set1_model_name_to_spec.keys()),
        in_csv="pcp_inputs/{pcp_input}.csv",
        hdf5_path="pcp_inputs/{pcp_input}.hdf5",
    output:
        touch("{pcp_input}_{model_name}.done"),
        aaprob="output/{pcp_input}/set2/{model_name}/aaprob.hdf5",
        performance="output/{pcp_input}/set2/{model_name}/performance.csv",
    params:
        model_class=lambda wildcards: set2_model_name_to_spec[wildcards.model_name][0],
        model_params=lambda wildcards: set2_model_name_to_spec[wildcards.model_name][1],
    benchmark:
        "output/{pcp_input}/set2/{model_name}/timing.tsv"
    shell:
        """
        mkdir -p output/{wildcards.pcp_input}/set2/{wildcards.model_name}
        epam aaprob {params.model_class} '{params.model_params}' {input.in_csv} {output.aaprob} {input.hdf5_path}
        epam evaluate {output.aaprob} {output.performance}
        """


rule run_model_set3:
    input:
        expand("{{pcp_input}}_{model}.done", model = set2_model_name_to_spec.keys()),
        in_csv="pcp_inputs/{pcp_input}.csv",
        hdf5_path="pcp_inputs/{pcp_input}.hdf5",
    output:
        touch("{pcp_input}_{model_name}.done"),
        aaprob="output/{pcp_input}/set3/{model_name}/aaprob.hdf5",
        performance="output/{pcp_input}/set3/{model_name}/performance.csv",
    params:
        model_class=lambda wildcards: set3_model_name_to_spec[wildcards.model_name][0],
        model_params=lambda wildcards: set3_model_name_to_spec[wildcards.model_name][1],
    benchmark:
        "output/{pcp_input}/set3/{model_name}/timing.tsv"
    shell:
        """
        mkdir -p output/{wildcards.pcp_input}/set3/{wildcards.model_name}
        epam aaprob {params.model_class} '{params.model_params}' {input.in_csv} {output.aaprob} {input.hdf5_path}
        epam evaluate {output.aaprob} {output.performance}
        """


rule clean_touch_files:
    input:
        expand("{{pcp_input}}_{model}.done", model = model_name_to_spec.keys())
    shell:
        """
        rm {pcp_input}_{model}.touch
        """

model_combos = ["set1/AbLang_heavy", "set1/ESM1v_default", "set2/SHMple_default", "set2/SHMple_productive", "set3/SHMple_ESM1v"]


rule combine_performance_files:
    input:
        expand(
            "output/{pcp_input}/{set_model}/performance.csv",
            pcp_input=pcp_inputs,
            set_model=model_combos,
        ),
    output:
        "output/combined_performance.csv",
        "output/combined_timing.csv",
    run:
        input_files = ",".join(input)
        input_timing_files = ",".join(
            f"output/{pcp_input}/{set_model}/timing.tsv"
            for pcp_input in pcp_inputs
            for set_model in model_combos
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

