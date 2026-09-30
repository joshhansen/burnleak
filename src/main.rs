use burn::{
    nn::{Linear, LinearConfig},
    tensor::{activation::sigmoid, backend::Backend, Device, Tensor},
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

fn leak<B: Backend>(device: &B::Device) {
    println!("Training on device: {:?}", device);

    let model: Model<B> = Model::init(device);

    loop {
        let x: Tensor<B, 2> = Tensor::ones([BATCH_SIZE, IN], device);
        let _v = model.forward(x);
    }
}

#[cfg(feature = "flex")]
fn main() {
    use burn::backend::{flex::FlexDevice, Flex};

    let device: Device<Flex> = FlexDevice;

    leak::<Flex>(&device);
}

#[cfg(feature = "wgpu")]
fn main() {
    use burn::backend::{wgpu::WgpuDevice, Wgpu};

    // let device: Device<Wgpu> = WgpuDevice::DiscreteGpu(0);
    let device: Device<Wgpu> = WgpuDevice::Cpu;

    leak::<Wgpu>(&device);
}
