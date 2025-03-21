from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_cold

estimate_zero3_model_states_mem_needs_all_cold(
    72e9, 1e9, num_gpus_per_node=8, num_nodes=1
)
