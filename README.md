# burnleak

Demonstrates a memory leak in the Wgpu backend of Burn

Branches:
* master: `cargo run --release` to see a steady memory leak
* master_ndarray: `cargo run --release` to see no memory leak

On `master`, setting the device as WgpuDevice::Cpu results in a much slower, but still existing leak. But the bulk of the
leak seems to step from the use of the discrete GPU.
