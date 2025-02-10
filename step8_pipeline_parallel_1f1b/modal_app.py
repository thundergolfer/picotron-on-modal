import os

import modal
import modal.experimental

# Instructions for install flash-attn taken from this Modal guide doc:
# https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .add_local_file("../pyproject.toml", "/pyproject.toml", copy=True)
    # flash-attn has an undeclared dependency on PyPi packages 'torch' and 'packaging',
    # as well as git, requiring that we annoyingly install without it the first time.
    #
    # ref: https://github.com/astral-sh/uv/issues/6437#issuecomment-2535324784
    .apt_install("git")
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    # --inexact is used to avoid removing the Modal client's dependencies.
    .run_commands(
        "uv sync --inexact --python-preference=only-system --no-install-package flash-attn",
        "uv sync --inexact --python-preference=only-system",
    )
)
app = modal.App("picotron-on-modal-step8-pipeline-parallel-1f1b", image=image)

# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
n_nodes = 4
# Typically this matches the number of GPUs per container.
n_proc_per_node = 4

LOCAL_CODE_DIR = "./"
REMOTE_CODE_DIR = "/root/"
REMOTE_SCRIPT_PATH = "/root/train.py"
GPU_TYPE = modal.gpu.H100(count=4)


@app.function(
    gpu=GPU_TYPE,
    mounts=[
        # Mount the script and other modules that performs the actual training.
        # Our modal.Function is merely a 'launcher' that sets up the distributed
        # cluster environment and then calls torch.distributed.run with desired arguments.
        modal.Mount.from_local_dir(
            LOCAL_CODE_DIR,
            remote_path=REMOTE_CODE_DIR,
        )
    ],
    secrets=[
        # Required for connecting to Weights & Biases from within the Modal container.
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=3600,
)
@modal.experimental.clustered(n_nodes)
def demo():
    from torch.distributed.run import parse_args, run

    cluster_info = modal.experimental.get_cluster_info()
    # which container am I?
    container_rank: int = cluster_info.rank
    # what's the leader/master/main container's address?
    main_ip_addr: str = cluster_info.container_ips[0]
    container_id = os.environ["MODAL_TASK_ID"]

    print(f"hello from {container_id}, rank {container_rank} of {n_nodes}")
    if container_rank == 0:
        print(f"main container's address: {main_ip_addr}")


    max_tokens = 16 * 1024 * 1024  # ~16M tokens

    args = [
        f"--nnodes={n_nodes}",
        f"--nproc-per-node={n_proc_per_node}",
        f"--node-rank={cluster_info.rank}",
        f"--master-addr={main_ip_addr}",
        REMOTE_SCRIPT_PATH,
        "--tp_size",
        "2",
        "--pp_size",
        "2",
        "--dp_size",
        "4",
        "--pp_engine",
        "1f1b",
        "--micro_batch_size",
        "4",
        "--gradient_accumulation_steps",
        "8",
        "--seq_len",
        "128",
        # Increasing max_tokens makes the model train more.
        "--max_tokens",
        str(max_tokens),
        "--num_proc",
        "16",
        "--run_name",
        "pp_1f1b",
        "--use_wandb",
    ]
    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))
