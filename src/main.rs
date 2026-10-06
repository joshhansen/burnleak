use burn::{
    nn::{Linear, LinearConfig},
    tensor::{activation::sigmoid, Device, Tensor},
};

const BATCH_SIZE: usize = 2048;

const OUT: usize = 50;

const IN: usize = 200;

struct Model {
    dense: Linear,
}

impl Model {
    fn init(device: &Device) -> Self {
        Self {
            dense: LinearConfig::new(IN, OUT).init(device),
        }
    }

    fn forward(&self, features: Tensor<2>) -> Tensor<2> {
        sigmoid(self.dense.forward(features))
    }
}

fn leak(device: &Device) {
    println!("Training on device: {:?}", device);

    let model: Model = Model::init(device);

    loop {
        let x: Tensor<2> = Tensor::ones([BATCH_SIZE, IN], device);
        let _v = model.forward(x);
    }
}

#[cfg(feature = "flex")]
fn main() {
    let device = Device::flex();

    leak(&device);
}

#[cfg(feature = "wgpu")]
fn main() {
    use burn::tensor::DeviceKind;

    // let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
    let device = Device::wgpu(DeviceKind::Cpu);

    leak(&device);
}
