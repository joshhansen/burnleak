use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig,
    },
    tensor::{
        activation::{relu, sigmoid},
        backend::Backend,
        Device, Distribution, Tensor,
    },
};
use num_traits::ToPrimitive;

const POSSIBLE_DIRECTIONS: usize = 8;

const POSSIBLE_UNIT_TYPES: usize = 10;

#[allow(non_camel_case_types)]
type fX = f32;

const POSSIBLE_CITY_ACTIONS: usize = POSSIBLE_UNIT_TYPES; // all possible productions

const POSSIBLE_UNIT_ACTIONS: usize = POSSIBLE_DIRECTIONS + 2; // plus skip and disband

const POSSIBLE_ACTIONS: usize = POSSIBLE_CITY_ACTIONS + POSSIBLE_UNIT_ACTIONS;

const DEEP_WIDTH: usize = 15;
const DEEP_HEIGHT: usize = 15;
const DEEP_TILES: usize = DEEP_WIDTH * DEEP_HEIGHT;

const DEEP_OUT_WIDTH: usize = 3;
const DEEP_OUT_HEIGHT: usize = 3;
const DEEP_OUT_TILES: usize = DEEP_OUT_WIDTH * DEEP_OUT_HEIGHT;

/// Number of "channels" in convolution output
const BASE_CONV_FEATS: usize = 20;

const DEEP_IN_LEN: usize = DEEP_TILES * BASE_CONV_FEATS;

const PER_ACTION_CHANNELS: usize = 1;

/// Total length of convolution output after reducing
const DEEP_OUT_LEN: usize = DEEP_OUT_TILES * POSSIBLE_ACTIONS * PER_ACTION_CHANNELS;

struct AgzActionModel<B: Backend> {
    convs: Vec<Conv2d<B>>,
    dense_common: Vec<Linear<B>>,
}

impl<B: Backend> AgzActionModel<B> {
    fn init(possible_actions: usize, device: &B::Device) -> AgzActionModel<B> {
        let channels = possible_actions * PER_ACTION_CHANNELS;

        let convs = vec![
            Conv2dConfig::new([BASE_CONV_FEATS, channels], [3, 3]).init(device), // -> 13x13
            Conv2dConfig::new([channels, channels], [3, 3]).init(device),        // -> 11x11
            Conv2dConfig::new([channels, channels], [3, 3]).init(device),        // -> 9x9
            Conv2dConfig::new([channels, channels], [3, 3]).init(device),        // -> 7x7
            Conv2dConfig::new([channels, channels], [3, 3]).init(device),        // -> 5x5
            Conv2dConfig::new([channels, channels], [3, 3]).init(device),        // -> 3x3
        ];

        let dense_common = vec![
            LinearConfig::new(DEEP_OUT_LEN, 64).init(device),
            LinearConfig::new(64, POSSIBLE_ACTIONS).init(device),
        ];

        AgzActionModel {
            convs,
            dense_common,
        }
    }

    fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        // Wide features that will pass through to the dense layers directly
        // [batch,wide_feat]
        let batches = features.dims()[0];

        // Input features to the 2d convolution
        // [batch,conv_feat,x,y]
        let mut deep = features.reshape([
            batches as i32,
            BASE_CONV_FEATS as i32,
            DEEP_HEIGHT as i32,
            DEEP_WIDTH as i32,
        ]);

        for conv in self.convs.iter() {
            deep = relu(conv.forward(deep));
        }

        // Reshape back to vector
        // [batch,deep_feat]
        let deep_flat: Tensor<B, 2> = deep.reshape([batches as i32, DEEP_OUT_LEN as i32]);

        let mut out_common = deep_flat;
        for d in &self.dense_common {
            out_common = d.forward(out_common);
        }

        let action_probs = out_common;

        debug_assert_eq!(action_probs.dims().len(), 2);
        debug_assert_eq!(action_probs.dims()[0], batches);
        debug_assert_eq!(action_probs.dims()[1], POSSIBLE_ACTIONS);

        sigmoid(action_probs)
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

    let model: AgzActionModel<Wgpu> = AgzActionModel::init(POSSIBLE_ACTIONS, &device);

    for _ in 0..usize::MAX {
        let x: Tensor<Wgpu, 2> = Tensor::random(
            [2048, DEEP_IN_LEN],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let _v = model.evaluate_tensors(x);
    }
}
