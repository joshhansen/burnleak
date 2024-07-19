use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig,
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

/// How many unit types there are, counting city as a unit type
const POSSIBLE_UNIT_TYPES_WRIT_LARGE: usize = POSSIBLE_UNIT_TYPES + 1;

#[allow(non_camel_case_types)]
type fX = f32;

const P_DROPOUT: f64 = 0.4;

const POSSIBLE_CITY_ACTIONS: usize = POSSIBLE_UNIT_TYPES; // all possible productions

const POSSIBLE_UNIT_ACTIONS: usize = POSSIBLE_DIRECTIONS + 2; // plus skip and disband

const POSSIBLE_ACTIONS: usize = POSSIBLE_CITY_ACTIONS + POSSIBLE_UNIT_ACTIONS;

const ADDED_WIDE_FEATURES: usize = 13;

/// Number of 1d (wide) features
/// Includes `POSSIBLE_UNIT_TYPES` twice: once for the unit type one-hot-encoded, once for the overall unit counts, plus one for city
const WIDE_LEN: usize = POSSIBLE_UNIT_TYPES_WRIT_LARGE + POSSIBLE_UNIT_TYPES + ADDED_WIDE_FEATURES;
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

/// Total length of the feature vectors that are input to the dnn
const FEATS_LEN: usize = WIDE_LEN + DEEP_IN_LEN;

struct AgzActionModel<B: Backend> {
    dropout: Dropout,
    convs: Vec<Conv2d<B>>,
    dense_common: Vec<Linear<B>>,
    dense_per_action: Vec<Vec<Linear<B>>>,
}

impl<B: Backend> AgzActionModel<B> {
    fn init(
        possible_actions: usize,
        dropout_config: DropoutConfig,
        device: &B::Device,
    ) -> AgzActionModel<B> {
        let dropout = dropout_config.init();

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
            LinearConfig::new(64, 32).init(device),
        ];

        let dense_per_action = (0..POSSIBLE_ACTIONS)
            .map(|_| {
                vec![
                    LinearConfig::new(32, 16).init(device),
                    LinearConfig::new(16, 8).init(device),
                    LinearConfig::new(8, 1).init(device),
                ]
            })
            .collect();

        AgzActionModel {
            dropout,
            convs,
            dense_common,
            dense_per_action,
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

        for (i, conv) in self.convs.iter().enumerate() {
            deep = relu(conv.forward(deep));
            if i < 3 {
                deep = self.dropout.forward(deep);
            }
        }

        // Reshape back to vector
        // [batch,deep_feat]
        let deep_flat: Tensor<B, 2> = deep.reshape([batches as i32, DEEP_OUT_LEN as i32]);

        let mut out_common = deep_flat;
        for d in &self.dense_common {
            out_common = d.forward(out_common);
        }

        let out: Vec<Tensor<B, 2>> = (0..POSSIBLE_ACTIONS)
            .map(|action_idx| {
                let mut out = out_common.clone();
                for (i, dense) in self.dense_per_action[action_idx].iter().enumerate() {
                    out = dense.forward(out);
                    // Only relu non-finally
                    if i < self.dense_per_action[action_idx].len() - 1 {
                        out = relu(out);
                    }
                }
                out
            })
            .collect();

        let action_probs = Tensor::cat(out, 1);

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

    let dropout_config = DropoutConfig { prob: P_DROPOUT };

    let model: AgzActionModel<Wgpu> =
        AgzActionModel::init(POSSIBLE_ACTIONS, dropout_config, &device);

    for _ in 0..usize::MAX {
        let x: Tensor<Wgpu, 2> = Tensor::random(
            [2048, DEEP_IN_LEN],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let _v = model.evaluate_tensors(x);
    }
}
