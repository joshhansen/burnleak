# burnleak

Demonstrates a memory leak in the Wgpu backend of Burn

`cargo build --profile release-final`

`./target/release-final/burnleak`

On my system (Debian 13 Linux, RTX 6000 Ada discrete GPU) after about 5 days this resulted in 17.9GB of resident and 24.3GB of virtual memory beint utilized by `burnleak`.
