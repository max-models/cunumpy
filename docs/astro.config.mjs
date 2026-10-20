// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/cunumpy/, with math, the API reference and a link check.
import { defineDocs, pydocsSidebarGroup } from '@max-models/starlight-theme';

const page = (/** @type {string} */ slug) => ({ slug });

export default defineDocs({
	repo: 'cunumpy',
	title: 'cunumpy',
	description: 'One Python code base on NumPy (CPU) or CuPy (NVIDIA GPU): replace import numpy as np with import cunumpy as xp.',
	pythonPackage: 'cunumpy',
	pydocs: {
		docstringStyle: 'numpy',
		members: {
			exclude: [
				// the maybempi stand-ins that cunumpy.mpi re-exports, documented in maybempi
				...['get_mpi', 'is_serial', 'launched_under_mpi', 'local_rank', 'OVERRIDE_VARIABLE', 'SerialMPI', 'SerialComm', 'SerialRequest', 'SerialStatus'].map(
					(name) => `cunumpy.mpi.${name}`
				),
			],
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Guides', link: '/guides/backends/' },
		{ label: 'Kernels', link: '/kernels/overview/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'API', link: '/reference/', section: '/api/' },
	],
	sidebar: [
		{ label: 'Getting started', items: ['getting-started/installation', 'getting-started/quickstart'].map(page) },
		{
			label: 'User guide',
			items: [
				'guides/backends',
				'guides/portable-code',
				'guides/data-movement',
				'guides/array-ordering',
				'guides/gpu-devices',
				'guides/execution-helpers',
				'guides/mpi',
				'guides/particle-codes',
				'guides/solvers',
				'guides/profiling',
				'guides/array-api-compat',
			].map(page),
		},
		{
			label: 'Porting kernels',
			items: [
				'kernels/overview',
				'kernels/pyccel-kernel',
				'kernels/cuda-kernel',
				'kernels/cuda-headers',
				'kernels/metal-kernel',
				'kernels/dispatch',
				'kernels/arguments',
				'kernels/accumulation',
				'kernels/debugging',
				'kernels/testing',
			].map(page),
		},
		{ label: 'Examples', items: [{ label: 'Overview', slug: 'examples' }, ...['examples/portable-script', 'examples/particle-pusher'].map(page)] },
		{ label: 'More', items: ['best-practices', 'troubleshooting', 'pyodide', 'ai-assistants'].map(page) },
		{
			label: 'Development',
			items: [
				{ label: 'Writing docstrings', slug: 'development/contributing' },
				{ label: 'Building the docs', slug: 'development/docs' },
			],
		},
		{
			label: 'API reference',
			items: [{ label: 'Modules', slug: 'reference' }, { label: 'API overview', slug: 'reference/overview' }, pydocsSidebarGroup],
		},
	],
	astro: {
		// the ai-assistants page renders src/cunumpy/LLM_GUIDE.md, outside docs/
		vite: { server: { fs: { allow: ['..'] } } },
		// the URLs of the former Sphinx site
		redirects: {
			'/ai-assistants.html': '/ai-assistants/',
			'/api.html': '/reference/overview/',
			'/array-api-compat.html': '/guides/array-api-compat/',
			'/best-practices.html': '/best-practices/',
			'/contributing.html': '/development/contributing/',
			'/examples/particle-pusher.html': '/examples/particle-pusher/',
			'/examples/portable-script.html': '/examples/portable-script/',
			'/guides/array-ordering.html': '/guides/array-ordering/',
			'/guides/backends.html': '/guides/backends/',
			'/guides/data-movement.html': '/guides/data-movement/',
			'/guides/execution-helpers.html': '/guides/execution-helpers/',
			'/guides/gpu-devices.html': '/guides/gpu-devices/',
			'/guides/mpi.html': '/guides/mpi/',
			'/guides/particle-codes.html': '/guides/particle-codes/',
			'/guides/portable-code.html': '/guides/portable-code/',
			'/guides/profiling.html': '/guides/profiling/',
			'/guides/solvers.html': '/guides/solvers/',
			'/installation.html': '/getting-started/installation/',
			'/kernels/accumulation.html': '/kernels/accumulation/',
			'/kernels/arguments.html': '/kernels/arguments/',
			'/kernels/cuda-headers.html': '/kernels/cuda-headers/',
			'/kernels/cuda-kernel.html': '/kernels/cuda-kernel/',
			'/kernels/debugging.html': '/kernels/debugging/',
			'/kernels/dispatch.html': '/kernels/dispatch/',
			'/kernels/metal-kernel.html': '/kernels/metal-kernel/',
			'/kernels/overview.html': '/kernels/overview/',
			'/kernels/pyccel-kernel.html': '/kernels/pyccel-kernel/',
			'/kernels/testing.html': '/kernels/testing/',
			'/pyodide.html': '/pyodide/',
			'/quickstart.html': '/getting-started/quickstart/',
			'/reference/algorithms.html': '/api/cunumpy/algorithms/',
			'/reference/arguments.html': '/api/cunumpy/arguments/',
			'/reference/cuda.html': '/api/cunumpy/cuda/',
			'/reference/integrate.html': '/api/cunumpy/integrate/',
			'/reference/kernel-testing.html': '/api/cunumpy/kernel_testing/',
			'/reference/kernels.html': '/api/cunumpy/kernels/',
			'/reference/memory.html': '/api/cunumpy/memory/',
			'/reference/mpi.html': '/api/cunumpy/mpi/',
			'/reference/optimize.html': '/api/cunumpy/optimize/',
			'/reference/petsc.html': '/api/cunumpy/petsc/',
			'/reference/profiling.html': '/api/cunumpy/profiling/',
			'/reference/rng.html': '/api/cunumpy/rng/',
			'/reference/top-level.html': '/api/cunumpy/',
			'/troubleshooting.html': '/troubleshooting/',
		},
	},
});
