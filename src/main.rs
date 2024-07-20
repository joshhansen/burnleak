use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    nn::{Linear, LinearConfig},
    tensor::{activation::sigmoid, backend::Backend, Device, Distribution, Tensor},
};

const BATCH_SIZE: usize = 2048;

const OUT: usize = 50;

const IN: usize = 200;

struct Model<B: Backend> {
    dense: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            dense: LinearConfig::new(IN, OUT).init(device),
        }
    }

    fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        sigmoid(self.dense.forward(features))
    }
}

fn main() {
    let device: Device<Wgpu> = WgpuDevice::DiscreteGpu(0);
    // let device: Device<Wgpu> = WgpuDevice::Cpu;

    let model: Model<Wgpu> = Model::init(&device);

    loop {
        let x: Tensor<Wgpu, 2> =
            Tensor::random([BATCH_SIZE, IN], Distribution::Uniform(0.0, 1.0), &device);
        let _v = model.forward(x);
    }
}
