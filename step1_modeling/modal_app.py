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
app = modal.App("picotron-on-modal-step1-modeling", image=image)

# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
n_nodes = 1
# Typically this matches the number of GPUs per container.
n_proc_per_node = 1

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
)
# Note that we don't yet need @modal.experimental.clustered because we're using only
# one node (a.k.a. container).
def demo():
    from torch.distributed.run import parse_args, run

    print("hello from modal")

    run(
        parse_args(
            [
                f"--nnodes={n_nodes}",
                f"--nproc-per-node={n_proc_per_node}",
                REMOTE_SCRIPT_PATH,
            ]
        )
    )
