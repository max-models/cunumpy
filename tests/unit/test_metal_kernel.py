"""MetalKernel: argument checking everywhere, launches on Apple silicon with MLX."""

import numpy as np
import pytest

import cunumpy as xp

MetalKernel = xp.kernels.MetalKernel
needs_metal = pytest.mark.skipif(
    not xp.kernels.metal_available(), reason="needs MLX and a Metal GPU"
)

AXPY = "uint i = thread_position_in_grid.x; y[i] = a[0] * x[i] + b[i];"


def axpy(**kwargs):
    return MetalKernel(AXPY, inputs=["x", "a", "b"], outputs=["y"], **kwargs)


def test_constructor_validates_names_and_options():
    with pytest.raises(ValueError, match="at least one output"):
        MetalKernel("", inputs=["x"], outputs=[])
    with pytest.raises(ValueError, match="distinct"):
        MetalKernel("", inputs=["x"], outputs=["x"])
    with pytest.raises(ValueError, match="float64"):
        MetalKernel("", inputs=["x"], outputs=["y"], float64="ignore")
    with pytest.raises(ValueError, match="threadgroup"):
        MetalKernel("", inputs=["x"], outputs=["y"], threadgroup=(1, 2, 3, 4))


def test_missing_mlx_or_gpu_gives_a_clear_error(monkeypatch):
    import cunumpy._metal_kernel as module

    def unavailable():
        raise ImportError("no mlx")

    monkeypatch.setattr(module, "_mlx", unavailable)
    assert not xp.kernels.metal_available()
    with pytest.raises(ImportError):
        axpy()(np.zeros(1, np.float32), 1.0, np.zeros(1, np.float32), out=np.zeros(1))


@needs_metal
def test_wrong_argument_counts_and_outputs_are_rejected():
    kernel = axpy()
    x = np.zeros(4, np.float32)
    with pytest.raises(TypeError, match="takes 3 input"):
        kernel(x, out=x.copy())
    with pytest.raises(TypeError, match="out must be 1 NumPy"):
        kernel(x, 1.0, x, out=[x, x])
    with pytest.raises(TypeError, match="out must be 1 NumPy"):
        kernel(x, 1.0, x, out=None)


@needs_metal
def test_float64_is_rejected_unless_cast():
    x = np.arange(4.0)
    with pytest.raises(TypeError, match="does not support"):
        axpy()(x.astype(np.float32), 2.0, x, out=np.empty(4, np.float32))
    with pytest.raises(TypeError, match="output 'y'"):
        axpy()(x.astype(np.float32), 2.0, x.astype(np.float32), out=np.empty(4))


@needs_metal
def test_axpy_with_python_scalar_and_non_contiguous_input():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    assert not x.flags.c_contiguous
    b = np.ones(x.shape, np.float32)
    y = np.empty(x.shape, np.float32)
    result = axpy()(x, 2.0, b, out=y, n_threads=x.size)
    assert result is y
    np.testing.assert_allclose(y, 2.0 * x + 1.0)


@needs_metal
def test_float64_cast_computes_in_float32_and_fills_float64_output():
    x = np.linspace(0, 1, 100)
    y = np.empty(100)
    axpy(float64="cast")(x, 3.0, x, out=y)
    np.testing.assert_allclose(y, 4.0 * x, rtol=1e-6)
    assert y.dtype == np.float64


@needs_metal
def test_several_outputs_template_and_init_value():
    kernel = MetalKernel(
        "uint i = thread_position_in_grid.x;"
        "if (i < N) { s[i] = x[i] + 1; d[i] = x[i] - 1; }",
        inputs=["x"],
        outputs=["s", "d"],
        init_value=-99.0,
    )
    x = np.arange(4, dtype=np.float32)
    s, d = np.empty(4, np.float32), np.empty(4, np.float32)
    out = kernel(x, out=(s, d), n_threads=4, template={"N": 3})
    assert out == (s, d)
    np.testing.assert_array_equal(s, [1, 2, 3, -99])
    np.testing.assert_array_equal(d, [-1, 0, 1, -99])


@needs_metal
def test_push_matches_float64_reference():
    n, steps, dt, B = 1000, 50, 0.01, 1.5
    c, s = np.cos(dt * B), np.sin(dt * B)
    rng = np.random.default_rng(0)
    pos = rng.random((n, 3)).astype(np.float32)
    vel = rng.standard_normal((n, 3)).astype(np.float32)
    kernel = MetalKernel(
        """
        uint i = thread_position_in_grid.x;
        float px = pos[3*i], py = pos[3*i+1], pz = pos[3*i+2];
        float vx = vel[3*i], vy = vel[3*i+1], vz = vel[3*i+2];
        for (int k = 0; k < NSTEPS; ++k) {
            px += vx*p[2]; py += vy*p[2]; pz += vz*p[2];
            float nx = p[0]*vx + p[1]*vy; vy = -p[1]*vx + p[0]*vy; vx = nx;
        }
        pos_out[3*i] = px; pos_out[3*i+1] = py; pos_out[3*i+2] = pz;
        vel_out[3*i] = vx; vel_out[3*i+1] = vy; vel_out[3*i+2] = vz;
        """,
        inputs=["pos", "vel", "p"],
        outputs=["pos_out", "vel_out"],
    )
    pos_out, vel_out = np.empty_like(pos), np.empty_like(vel)
    kernel(
        pos,
        vel,
        np.array([c, s, dt], np.float32),
        out=(pos_out, vel_out),
        n_threads=n,
        template={"NSTEPS": steps},
    )
    p, v = pos.astype(float), vel.astype(float)
    for _ in range(steps):
        p += v * dt
        v[:, :2] = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], 1)
    np.testing.assert_allclose(pos_out, p, atol=1e-4)
    np.testing.assert_allclose(vel_out, v, atol=1e-4)


@needs_metal
def test_copies_are_counted():
    x = np.zeros(8, np.float32)
    with xp.profiling.count_transfers() as counter:
        axpy()(x, 1.0, x, out=np.empty(8, np.float32))
    assert counter.to_device == 3 and counter.to_host == 1
