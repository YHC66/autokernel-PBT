"""Fixtures for CPU tests of device-side modules.

`kernels/triton/ladder.py` and the mutant modules import torch and triton at module
scope, so on a machine with neither they cannot even be imported. Their *structure* --
tile-width derivation, launcher guards, registration metadata -- is still CPU-checkable,
because none of it calls into either library until a kernel is actually launched.
"""

from __future__ import annotations

import sys
import types

import pytest

#: Top-level names whose modules are replaced (or, for `kernels`, re-imported against
#: the replacements) for the duration of one test.
_STUBBED_ROOTS = ("torch", "triton", "kernels")


def _is_stubbed(name: str) -> bool:
    return any(name == root or name.startswith(f"{root}.") for root in _STUBBED_ROOTS)


@pytest.fixture
def device_stubs():
    """Install a bare `torch` and a `triton` whose `jit` is the identity.

    Restores `sys.modules` exactly afterwards, including on a machine that HAS the real
    libraries: a `kernels.triton.ladder` left bound to the stub would be picked up by a
    later device test in the same session and fail there for no reason of its own.
    """
    saved = {name: module for name, module in sys.modules.items() if _is_stubbed(name)}
    for name in saved:
        del sys.modules[name]

    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    triton.jit = lambda fn: fn
    triton.language = language
    torch = types.ModuleType("torch")
    sys.modules.update({"triton": triton, "triton.language": language, "torch": torch})
    try:
        yield types.SimpleNamespace(triton=triton, torch=torch)
    finally:
        for name in [name for name in sys.modules if _is_stubbed(name)]:
            del sys.modules[name]
        sys.modules.update(saved)


class FakeToolchain:
    """A `CudaToolchain` that compiles and launches nothing, and records every request.

    `fail_compile` makes `compile` raise it; `artifact` overrides what a compile returns.
    The default artifact carries a value for every attribute a real one has, so a test
    that finds a CUDA field MISSING learns something about the code under test, not about
    the fake.
    """

    ARCH_OPTION = "--gpu-architecture=compute_86"

    def __init__(self, fail_compile: Exception | None = None, artifact=None, versions=None):
        self.fail_compile = fail_compile
        self.artifact = artifact
        self._versions = versions
        self.compiles: list[tuple[str, str, tuple[str, ...]]] = []
        self.launches: list[dict] = []

    def resolved_options(self, options):
        return (*options, self.ARCH_OPTION)

    def compile(self, source, entry, options):
        from autokernel_pbt.props.backends.cuda_toolchain import CudaArtifact

        self.compiles.append((source, entry, tuple(options)))
        if self.fail_compile is not None:
            raise self.fail_compile
        if self.artifact is not None:
            return self.artifact
        return CudaArtifact(
            entry=entry,
            options=self.resolved_options(tuple(options)),
            asm={"ptx": f"// ptx for {entry}\n{source}"},
            # Every value distinct, so a field read from the wrong attribute cannot pass.
            n_regs=24,
            static_shared_bytes=48,
            local_bytes=16,
            max_threads_per_block=1024,
            ptx_version=80,
            binary_version=86,
        )

    def launch(self, artifact, *, grid, block, shared_bytes, args):
        self.launches.append(
            {"artifact": artifact, "grid": grid, "block": block, "shared_bytes": shared_bytes,
             "args": args}
        )

    def versions(self):
        if self._versions is not None:
            return self._versions()
        return {"nvrtc_version": "12.8", "cuda_bindings_version": "12.8.0"}


@pytest.fixture
def fake_toolchain():
    """The `FakeToolchain` class; tests construct it with the behaviour they need."""
    return FakeToolchain
