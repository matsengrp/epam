import epam.models
import json
import subprocess

# ====== User settings ======
number_of_batches = 5
pcp_per_batch = 20
# ============

model_name_to_spec = {
    model_name: [model_class, json.dumps({**model_params, "model_name": model_name})]
    for model_name, model_class, model_params in epam.models.FULLY_SPECIFIED_MODELS 
}

set1_models = ("AbLang_heavy", "ESM1v_default")
set2_models = ("SHMple_default", "SHMple_productive")
set3_models = "SHMple_ESM1v"

model_combos = ["set1/AbLang_heavy", "set1/ESM1v_default", "set2/SHMple_default", "set2/SHMple_productive", "set3/SHMple_ESM1v"]

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
batch_number = range(1, number_of_batches+1)

def get_model_class(model_name, set_model_name_to_spec):
    return set_model_name_to_spec.get(model_name, (None, None))[0]

def get_model_params(model_name, set_model_name_to_spec):
    return set_model_name_to_spec.get(model_name, (None, None))[1]


rule all:
    input:
        "output/combined_performance.csv",
        "output/combined_timing.csv",


rule split_pcp_batches:
    input:
        "pcp_inputs/{pcp_input}.csv"
    output:
        out_csvs=dynamic("pcp_batched_inputs/{pcp_input}_{part}.csv")  
    params:
        output_dir="pcp_batched_inputs/",
        batch_size=pcp_per_batch
    shell:
        """
        python scripts/split_pcp_files.py {input} {params.output_dir} {params.batch_size}
        """


rule precompute_esm:
    input:
        in_csv="pcp_batched_inputs/{pcp_input}_{part}.csv", 
    output:
        out_hdf5="pcp_batched_inputs/{pcp_input}_{part}.hdf5", 
    params:
        part=lambda wildcards: wildcards.part  # Define a dynamic wildcard for {part}
        strategy="masked-marginals"
    shell:
        """
        epam esm_bulk_precompute {input.in_csv} {output.out_hdf5} {params.strategy}
        """


rule run_model_set1:
    input:
        in_csv="pcp_batched_inputs/{pcp_input}_{part}.csv",
        hdf5_path="pcp_batched_inputs/{pcp_input}_{part}.hdf5",
    output:
        complete="_ignore/flag_files/{pcp_input}_{part}_{model_name}.done",
        aaprob="output/{pcp_input}/set1/{model_name}/batch{part}/aaprob.hdf5",
    params:
        part=lambda wildcards: wildcards.part,
        model_class=lambda wildcards: get_model_class(wildcards.model_name, set1_model_name_to_spec),
        model_params=lambda wildcards: get_model_params(wildcards.model_name, set1_model_name_to_spec),
    benchmark:
        "output/{pcp_input}/set1/{model_name}/batch{part}/timing.tsv"
    wildcard_constraints:
        model_name="|".join(set1_models),
    shell:
        """
        mkdir -p output/{wildcards.pcp_input}/set1/{wildcards.model_name}/batch{params.part}
        epam aaprob {params.model_class} '{params.model_params}' {input.in_csv} {output.aaprob} {input.hdf5_path}
        touch {output.complete}
        """


rule run_model_set2:
    input:
        expand("_ignore/flag_files/{pcp_input}_{part}_{model}.done", pcp_input=pcp_inputs, part=batch_number, model=set1_model_name_to_spec.keys()),
        in_csv="pcp_batched_inputs/{pcp_input}_{part}.csv",
        hdf5_path="pcp_batched_inputs/{pcp_input}_{part}.hdf5",
    output:
        complete="_ignore/flag_files/{pcp_input}_{part}_{model_name}.done",
        aaprob="output/{pcp_input}/set2/{model_name}/batch{part}/aaprob.hdf5",
    params:
        part=lambda wildcards: wildcards.part,
        model_class=lambda wildcards: get_model_class(wildcards.model_name, set2_model_name_to_spec),
        model_params=lambda wildcards: get_model_params(wildcards.model_name, set2_model_name_to_spec),
    benchmark:
        "output/{pcp_input}/set2/{model_name}/batch{part}/timing.tsv"
    wildcard_constraints:
        model_name="|".join(set2_models),
    shell:
        """
        mkdir -p output/{wildcards.pcp_input}/set2/{wildcards.model_name}/batch{params.part}
        epam aaprob {params.model_class} '{params.model_params}' {input.in_csv} {output.aaprob} {input.hdf5_path}
        touch {output.complete}
        """


rule run_model_set3:
    input:
        expand("_ignore/flag_files/{{pcp_input}}_{{part}}_{model}.done", pcp_input=pcp_inputs, part=batch_number, model = set2_model_name_to_spec.keys()),
        in_csv="pcp_batched_inputs/{pcp_input}_{part}.csv",
        hdf5_path="pcp_batched_inputs/{pcp_input}_{part}.hdf5",
    output:
        aaprob="output/{pcp_input}/set3/{model_name}/batch{part}/aaprob.hdf5",
    params:
        part=lambda wildcards: wildcards.part,
        model_class=lambda wildcards: get_model_class(wildcards.model_name, set3_model_name_to_spec),
        model_params=lambda wildcards: get_model_params(wildcards.model_name, set3_model_name_to_spec),
    benchmark:
        "output/{pcp_input}/set3/{model_name}/batch{part}/timing.tsv"
    wildcard_constraints:   
        model_name=set3_models,
    shell:
        """
        mkdir -p output/{wildcards.pcp_input}/set3/{wildcards.model_name}/batch{params.part}
        epam aaprob {params.model_class} '{params.model_params}' {input.in_csv} {output.aaprob} {input.hdf5_path}
        """


rule combine_aaprob_files:
    input:
        expand(
            "output/{{pcp_input}}/{{set_model}}/batch{part}/aaprob.hdf5",
            part=batch_number,
        ),
    output:
        "output/{pcp_input}/{set_model}/combined_aaprob.hdf5",
    run:
        input_files = ",".join(input)
        output_file = output[0]
        
        subprocess.run(
            f"epam concatenate_hdf5s {input_files} {output_file}", shell=True, check=True
        )


rule evaluate_performance:
    input:
        "output/{pcp_input}/{set_model}/combined_aaprob.hdf5", 
    output:
        "output/{pcp_input}/{set_model}/performance.csv",
    shell:
        """
        epam evaluate {input} {output}
        """


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
            f"output/{pcp_input}/{set_model}/batch{part}/timing.tsv"
            for pcp_input in pcp_inputs
            for set_model in model_combos
            for part in batch_number
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


rule clean_flag_files:
    shell:
        """
        rm -f _ignore/flag_files/*.done
        """
