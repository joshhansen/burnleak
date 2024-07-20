use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    nn::{Linear, LinearConfig},
    tensor::{activation::sigmoid, backend::Backend, Device, Distribution, Tensor},
};
use num_traits::ToPrimitive;

const POSSIBLE_DIRECTIONS: usize = 8;

const POSSIBLE_UNIT_TYPES: usize = 10;

#[allow(non_camel_case_types)]
type fX = f32;

const POSSIBLE_CITY_ACTIONS: usize = POSSIBLE_UNIT_TYPES; // all possible productions

const POSSIBLE_UNIT_ACTIONS: usize = POSSIBLE_DIRECTIONS + 2; // plus skip and disband

const POSSIBLE_ACTIONS: usize = POSSIBLE_CITY_ACTIONS + POSSIBLE_UNIT_ACTIONS;

const DEEP_OUT_WIDTH: usize = 3;
const DEEP_OUT_HEIGHT: usize = 3;
const DEEP_OUT_TILES: usize = DEEP_OUT_WIDTH * DEEP_OUT_HEIGHT;

const PER_ACTION_CHANNELS: usize = 1;

/// Total length of convolution output after reducing
const DEEP_OUT_LEN: usize = DEEP_OUT_TILES * POSSIBLE_ACTIONS * PER_ACTION_CHANNELS;

struct AgzActionModel<B: Backend> {
    dense: Linear<B>,
}

impl<B: Backend> AgzActionModel<B> {
    fn init(device: &B::Device) -> Self {
        AgzActionModel {
            dense: LinearConfig::new(DEEP_OUT_LEN, POSSIBLE_ACTIONS).init(device),
        }
    }

    fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let batches = features.dims()[0];

        let mut out = features;

        out = self.dense.forward(out);

        debug_assert_eq!(out.dims().len(), 2);
        debug_assert_eq!(out.dims()[0], batches);
        debug_assert_eq!(out.dims()[1], POSSIBLE_ACTIONS);

        sigmoid(out)
    }

    /// [batch,feat]
    fn evaluate_tensors(&self, features: Tensor<B, 2>) -> Vec<fX> {
        let result_tensor = self.forward(features);

        result_tensor
            .into_data()
            .value
            .into_iter()
            .map(|x| x.to_f32().unwrap())
            .collect()
    }
}

fn main() {
    let device: Device<Wgpu> = WgpuDevice::DiscreteGpu(0);

    let model: AgzActionModel<Wgpu> = AgzActionModel::init(&device);

    loop {
        let x: Tensor<Wgpu, 2> = Tensor::random(
            [2048, DEEP_OUT_LEN],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let _v = model.evaluate_tensors(x);
    }
}
