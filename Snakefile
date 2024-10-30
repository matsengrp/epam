import epam.models
import subprocess

# Read parameters from the config file
dataset = config["dataset"]
number_of_batches = config["number_of_batches"]
pcp_per_batch = config["pcp_per_batch"]
pcp_input = config["pcp_input"]

model_name_to_spec = {
    model_name: [model_class, str({**model_params, "model_name": model_name})]
    for model_name, model_class, model_params in epam.models.FULLY_SPECIFIED_MODELS 
}

# set1_models = ("AbLang1", "AbLang2_wt", "AbLang2_mask", "ESM1v_mask", "S5F", "S5FESM_mask", "S5FBLOSUM", "NetamSHM", "NetamSHM_productive", "NetamESM_mask", "NetamBLOSUM")

# model_combos = ["set1/AbLang1", "set1/AbLang2_wt", "set1/AbLang2_mask", "set1/ESM1v_mask", "set1/S5F", "set1/S5FESM_mask", "set1/S5FBLOSUM", "set1/NetamSHM", "set1/NetamSHM_productive", "set1/NetamESM_mask", "set1/NetamBLOSUM"]

# set1_model_name_to_spec = {
#     key: model_name_to_spec[key] for key in set1_models
# }

batch_number = range(1, number_of_batches+1)
esm_model_number = 1

# def get_model_class(model_name, set_model_name_to_spec):
#     return set_model_name_to_spec.get(model_name, (None, None))[0]

# def get_model_params(model_name, set_model_name_to_spec):
#     return set_model_name_to_spec.get(model_name, (None, None))[1]


rule all:
    input:
        expand("output/{pcp_input}/combined_performance.csv", pcp_input=config["pcp_input"]),
        expand("output/{pcp_input}/combined_timing.csv", pcp_input=config["pcp_input"]),


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
        out_hdf5="pcp_batched_inputs/{pcp_input}_esm{esm_model_number}_mask_logits_{part}.hdf5", 
    params:
        part=lambda wildcards: wildcards.part,  # Define a dynamic wildcard for {part}
        esm_model_number=lambda wildcards: wildcards.esm_model_number
    shell: 
        """
        epam esm_bulk_precompute {input.in_csv} {output.out_hdf5} "masked-marginals" {params.esm_model_number}
        """

rule process_esm:
    input:
        in_hdf5="pcp_batched_inputs/{pcp_input}_esm{esm_model_number}_mask_logits_{part}.hdf5",
    output:
        out_hdf5="pcp_batched_inputs/{pcp_input}_esm{esm_model_number}_mask_ratios_{part}.hdf5",
    params:
        part=lambda wildcards: wildcards.part,
        esm_model_number=lambda wildcards: wildcards.esm_model_number
    shell:
        """
        epam process_esm_output {input.in_hdf5} {output.out_hdf5} "masked-marginals"
        """


rule run_models:
    input:
        in_csv="pcp_batched_inputs/{pcp_input}_{part}.csv",
        hdf5_path="pcp_batched_inputs/{pcp_input}_esm{esm_model_number}_mask_ratios_{part}.hdf5",
    output:
        aaprob="output/{pcp_input}/{model_name}/batch{part}/aaprob.hdf5",
    params:
        part=lambda wildcards: wildcards.part,
        model_class=lambda wildcards: model_name_to_spec[wildcards.model_name][0],
        model_params=lambda wildcards: model_name_to_spec[wildcards.model_name][1],
        esm_model_number=lambda wildcards: wildcards.esm_model_number
        # model_class=lambda wildcards: get_model_class(wildcards.model_name, set1_model_name_to_spec),
        # model_params=lambda wildcards: get_model_params(wildcards.model_name, set1_model_name_to_spec),
    benchmark:
        "output/{pcp_input}/{model_name}/batch{part}/timing.tsv"
    # wildcard_constraints:
    #     model_name="|".join(set1_models),
    shell:
        """
        mkdir -p output/{wildcards.pcp_input}/{wildcards.model_name}/batch{params.part}
        epam aaprob {params.model_class} "{params.model_params}" {input.in_csv} {output.aaprob} {input.hdf5_path}
        """


rule combine_aaprob_files:
    input:
        expand(
            "output/{{pcp_input}}/{model_name}/batch{part}/aaprob.hdf5",
            model_name=model_name_to_spec.keys(),
            part=batch_number,
        ),
    output:
        "output/{pcp_input}/{model_name}/combined_aaprob.hdf5",
    run:
        input_files = ",".join(input)
        output_file = output[0]
        
        subprocess.run(
            f"epam concatenate_hdf5s {input_files} {output_file}", shell=True, check=True
        )


rule evaluate_performance:
    input:
        "output/{pcp_input}/{model_name}/combined_aaprob.hdf5", 
    output:
        "output/{pcp_input}/{model_name}/performance.csv",
    shell:
        """
        epam evaluate {input} {output}
        """


rule combine_performance_files:
    input:
        expand(
            "output/{pcp_input}/{model_name}/performance.csv",
            pcp_input=pcp_input,
            model_name=model_name_to_spec.keys(),
        ),
    output:
        "output/{pcp_input}/combined_performance.csv",
        "output/{pcp_input}/combined_timing.csv",
    run:
        input_files = ",".join(input)
        input_timing_files = ",".join(
            f"output/{pcp_input}/{model_name}/batch{part}/timing.tsv"
            for model_name in model_name_to_spec.keys()
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
