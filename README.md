# burnleak

Demonstrates a memory leak in the Wgpu backend of Burn, running on `WgpuDevice::DiscreteGpu(0)` by default:

`cargo build --profile release-final`

`./target/release-final/burnleak`

On my system (Debian 13 Linux, RTX 6000 Ada discrete GPU) after about 5 days this resulted in 17.9GB of resident and 24.3GB of virtual memory being utilized by `burnleak`.

For comparison, run it on the Flex backend using:

`cargo build --profile release-final --no-default-features -F flex`

`./target/release-final/burnleak`

If you run this alongside the one running on GPU, you should see fairly quickly a stark difference in system memory consumption. The one on GPU should grow without bound, if slowly. Try watching for 15 or 30 minutes and you should see it.
