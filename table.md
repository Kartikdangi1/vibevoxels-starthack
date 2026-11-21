| # | Phase (`key`) | Category | Official Baseline (`44.204 s`) | Before Today's Fixes (`18.557 s`) | After Our Optimizations (`5.669 s`) |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `imports` (`import jax, numpy, differometor`) | `setup` | `2.613 s` | `2.627 s` | `3.595 s` |
| 2 | `initialize` (`jax.devices("cpu")`) | `setup` | `0.017 s` | `0.015 s` | `0.034 s` |
| 3 | `load` (read & parse JSON fixtures) | `setup` | `0.004 s` | `0.004 s` | `0.004 s` |
| 4 | `build` (`run_build_step` + `compress_arrays`) | `setup` | `1.026 s` | `0.785 s` | `0.197 s` |
| 5 | `forward_lower` (trace & lower forward + VJP) | `compile` | `2.011 s` | `2.151 s` | `0.244 s` |
| 6 | `forward_compile` (XLA compile forward) | `compile` | `14.950 s` | `5.937 s` | `0.011 s` |
| 7 | `forward` (1st forward execution + AD residuals) | `execute` | `13.285 s` | `4.380 s` | `0.384 s` |
| 8 | `sensitivity_lower` (trace & lower readout VJP) | `compile` | `0.111 s` | `0.055 s` | `0.188 s` |
| 9 | `sensitivity_compile` (XLA compile readout VJP) | `compile` | `0.291 s` | `0.139 s` | `0.442 s` |
| 10 | `sensitivity` (1st readout execution) | `execute` | `0.001 s` | `0.001 s` | `0.002 s` |
| 11 | `gradient_lower` (trace & lower backward pass) | `compile` | `0.371 s` | `0.239 s` | `0.037 s` |
| 12 | `gradient_compile` (XLA compile backward pass) | `compile` | `5.184 s` | `2.145 s` | `0.232 s` |
| 13 | `gradient` (1st reverse-mode gradient execution) | `execute` | `4.339 s` | `0.099 s` | `0.299 s` |
| 14 | `host_transfer` (`jax.device_get`) | `output` | `0.0004 s` | `0.001 s` | `0.002 s` |
|  | Subtotal (Excluding `imports`) |  | `41.591 s` | `15.930 s` | `2.074 s` (`20.1x` faster) |
|  | Complete First-Run Total (All Phases) |  | `44.204 s` | `18.557 s` | `5.669 s` (`7.8x` faster) |
|  | Warm Repeat (`forward + sensitivity + gradient`) |  | `17.625 s` | `0.460 s` | `0.475 s` (`37.1x` faster) |