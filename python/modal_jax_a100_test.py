import subprocess

import modal


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("jax[cuda13]", "numpy")
)

app = modal.App("jax-a100-smoke-test", image=image)


@app.function(gpu="A100", timeout=600)
def run_jax_gpu_test() -> dict:
    import jax
    import jax.numpy as jnp

    print("nvidia-smi:")
    subprocess.run(["nvidia-smi"], check=False)

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4096, 4096), dtype=jnp.float32)

    @jax.jit
    def matmul_sum(a):
        return jnp.sum(a @ a.T)

    result = matmul_sum(x).block_until_ready()
    result_float = float(result)

    print("Result:", result_float)
    print("Result device:", result.device)

    return {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "result": result_float,
        "result_device": str(result.device),
    }


@app.local_entrypoint()
def main():
    output = run_jax_gpu_test.remote()
    print("Remote output:")
    print(output)
