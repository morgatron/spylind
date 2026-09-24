import numpy as np
import sympy as sm
import time
import gc
import os
import psutil
from spylind import spyIVP as so

def get_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

def run_single_benchmark(ode_sys, backend, t_steps, solver_opts=None):
    print(f"  Testing {backend}...")
    
    # Force GC before start
    gc.collect()
    
    try:
        # 1. Setup
        start_setup = time.time()
        model = ode_sys.setup_model(backend=backend, solver_opts=solver_opts)
        setup_time = time.time() - start_setup
        
        # 2. Warm-up (Compilation)
        model.integrate(t_steps[:2])
        
        # 3. Actual Benchmark
        mem_start = get_mem()
        start_time = time.time()
        res = model.integrate(t_steps)
        # For JAX, we might need to block until results are ready
        if hasattr(res, 'block_until_ready'):
            res.block_until_ready()
            
        duration = time.time() - start_time
        mem_end = get_mem()
        
        print(f"    DEBUG: {backend} result shape: {res.shape}, dtype: {res.dtype}")
        print(f"    DEBUG: {backend} first few results: {res[-1, :, 0]}")
        
        return {
            'status': 'success',
            'duration': duration,
            'setup_time': setup_time,
            'mem_delta': mem_end - mem_start,
            'result_final': res[-1],
            'dtype': res.dtype
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}

def benchmark_suite(n_systems_list=[1000, 5000, 10000], n_steps=100):
    print(f"Starting Benchmark Suite (steps={n_steps})")
    print("-" * 60)
    
    # Define problem
    deltaS, omegaS, gammaS = sm.symbols('delta, Omega, gamma')
    r00, r11, r01r, r01i = sm.symbols('r00, r11, r01r, r01i', real=True)
    
    dy_dtD = {
        r00: gammaS * r11 + omegaS * r01i,
        r11: -gammaS * r11 - omegaS * r01i,
        r01r: -deltaS * r01i - 0.5 * gammaS * r01r,
        r01i: deltaS * r01r - 0.5 * gammaS * r01i - 0.5 * omegaS * (r00 - r11)
    }
    
    t_steps = np.linspace(0, 50, n_steps)
    backends = ['numpy', 'diffrax', 'cyrk']
    solver_opts = {'max_steps': 1000000, 'atol': 1e-8, 'rtol': 1e-6}
    
    for n in n_systems_list:
        print(f"\nProblem Size: {n} systems ({n*4} variables)")
        
        deltas = np.linspace(-10, 10, n)
        ode_sys = so.ODESys(dy_dtD, trans_dims={deltaS: deltas})
        ode_sys.set_constants({omegaS: 1.0, gammaS: 0.1})
        ode_sys.set_initial_state({r00: 1.0, r11: 0.0, r01r: 0.0, r01i: 0.0})
        
        all_res = {}
        for b in backends:
            res = run_single_benchmark(ode_sys, b, t_steps, solver_opts)
            all_res[b] = res
            
        # Compare results
        valid_backends = [b for b in backends if all_res[b]['status'] == 'success']
        if len(valid_backends) > 1:
            ref = valid_backends[0]
            for other in valid_backends[1:]:
                # result_final shape: (n_vars, n_systems) or similar
                # Check mean absolute difference
                diff = np.mean(np.abs(all_res[ref]['result_final'] - all_res[other]['result_final']))
                print(f"    Check vs {ref}: {other} avg diff = {diff:.2e}")

        # Summary Table
        print(f"\n    {'Backend':10} | {'Time (s)':10} | {'Mem (MB)':10} | {'Setup (s)':10}")
        print("    " + "-" * 46)
        for b in backends:
            r = all_res[b]
            if r['status'] == 'success':
                print(f"    {b:10} | {r['duration']:10.4f} | {r['mem_delta']:10.2f} | {r['setup_time']:10.4f}")
            else:
                print(f"    {b:10} | {'FAILED':10} | {'-':10} | {'-':10}")

if __name__ == "__main__":
    # Enable JAX x64 if possible
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
    except:
        pass
        
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="1000,5000,10000")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    
    sizes = [int(s) for s in args.sizes.split(",")]
    benchmark_suite(n_systems_list=sizes, n_steps=args.steps)
