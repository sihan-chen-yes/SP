import numpy as np
import torch
import torch.nn as nn


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    if multires == 0:
        return lambda x: x, input_dims
    assert multires > 0

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class HannwEmbedder:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)

        # get hann window weights
        if self.cfg.full_band_iter <= 0 or self.cfg.kick_in_iter >= self.cfg.full_band_iter:
            alpha = torch.tensor(N_freqs, dtype=torch.float32)
        else:
            kick_in_iter = torch.tensor(self.cfg.kick_in_iter,
                                        dtype=torch.float32)
            t = torch.clamp(self.kwargs['iter_val'] - kick_in_iter, min=0.)
            N = self.cfg.full_band_iter - kick_in_iter
            m = N_freqs
            alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1. - torch.cos(np.pi * torch.clamp(alpha - freq_idx,
                                                    min=0., max=1.))) / 2.
            # print("freq_idx: ", freq_idx, "weight: ", w, "iteration: ", self.kwargs['iter_val'])
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_hannw_embedder(cfg, multires, iter_val,):
    embed_kwargs = {
        'include_input': False,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'periodic_fns': [torch.sin, torch.cos],
        'iter_val': iter_val
    }

    embedder_obj = HannwEmbedder(cfg, **embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class HierarchicalPoseEncoder(nn.Module):
    '''Hierarchical encoder from LEAP.'''

    def __init__(self, num_joints=24, rel_joints=False, dim_per_joint=6, out_dim=-1, **kwargs):
        super().__init__()

        self.num_joints = num_joints
        self.rel_joints = rel_joints
        self.ktree_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
            9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)

        self.layer_0 = nn.Linear(9*num_joints + 3*num_joints, dim_per_joint)
        dim_feat = 13 + dim_per_joint

        layers = []
        for idx in range(num_joints):
            layer = nn.Sequential(nn.Linear(dim_feat, dim_feat), nn.ReLU(), nn.Linear(dim_feat, dim_per_joint))

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if out_dim <= 0:
            self.out_layer = nn.Identity()
            self.n_output_dims = num_joints * dim_per_joint
        else:
            self.out_layer = nn.Linear(num_joints * dim_per_joint, out_dim)
            self.n_output_dims = out_dim

    def forward(self, rots, Jtrs, skinning_weight=None):
        batch_size = rots.size(0)

        if self.rel_joints:
            with torch.no_grad():
                Jtrs_rel = Jtrs.clone()
                Jtrs_rel[:, 1:, :] = Jtrs_rel[:, 1:, :] - Jtrs_rel[:, self.ktree_parents[1:], :]
                Jtrs = Jtrs_rel.clone()

        global_feat = torch.cat([rots.view(batch_size, -1), Jtrs.view(batch_size, -1)], dim=-1)
        global_feat = self.layer_0(global_feat)
        # global_feat = (self.layer_0.weight@global_feat[0]+self.layer_0.bias)[None]
        out = [None] * self.num_joints
        for j_idx in range(self.num_joints):
            rot = rots[:, j_idx, :]
            Jtr = Jtrs[:, j_idx, :]
            parent = self.ktree_parents[j_idx]
            if parent == -1:
                bone_l = torch.norm(Jtr, dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, global_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)
            else:
                parent_feat = out[parent]
                bone_l = torch.norm(Jtr if self.rel_joints else Jtr - Jtrs[:, parent, :], dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, parent_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)

        out = torch.cat(out, dim=-1)
        out = self.out_layer(out)
        return out

class VanillaCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, config, dim_coord=3):
        super(VanillaCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config["n_neurons"], config["n_hidden_layers"]

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if config["multires"] > 0:
            embed_fn, input_ch = get_embedder(config["multires"], input_dims=dim_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.last_layer_init = config.get('last_layer_init', False)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in config["skip_in"]:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config["cond_in"]:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if self.last_layer_init and l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=0., std=1e-5)
                torch.nn.init.constant_(lin.bias, val=0.)


            setattr(self, "lin" + str(l), lin)

        self.activation = nn.LeakyReLU()

    def forward(self, coords, cond=None):
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        if self.embed_fn is not None:
            coords_embedded = self.embed_fn(coords)
        else:
            coords_embedded = coords

        x = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config["cond_in"]:
                x = torch.cat([x, cond], 1)

            if l in self.config["skip_in"]:
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

def get_skinning_mlp(n_input_dims, n_output_dims, config):
    if config.otype == 'VanillaMLP':
        network = VanillaCondMLP(n_input_dims, 0, n_output_dims, config)
    else:
        raise ValueError

    return network


class HannwCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, config, dim_coord=3):
        super(HannwCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if config.multires > 0:
            _, input_ch = get_hannw_embedder(config.embedder, config.multires, 0)
            dims[0] = input_ch

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in config.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config.cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if l in config.cond_in:
                # Conditional input layer initialization
                torch.nn.init.constant_(lin.weight[:, -dim_cond:], 0.0)
            torch.nn.init.constant_(lin.bias, 0.0)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, coords, iteration, cond=None):
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        if self.config.multires > 0:
            embed_fn, _ = get_hannw_embedder(self.config.embedder, self.config.multires, iteration)
            coords_embedded = embed_fn(coords)
        else:
            coords_embedded = coords

        x = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config.cond_in:
                x = torch.cat([x, cond], 1)

            if l in self.config.skip_in:
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

# JOINT_NAMES = [
#     'pelvis', 0

#     'left_hip', 1
#     'right_hip',2
#     'spine1',3

#     'left_knee',4
#     'right_knee',5
#     'spine2',6

#     'left_ankle',7
#     'right_ankle',8
#     'spine3',9

#     'left_foot',10
#     'right_foot',11

#     'neck',12
#     'left_collar',13
#     'right_collar',14

#     'head',15

#     'left_shoulder',16
#     'right_shoulder',17
#     'left_elbow',18
#     'right_elbow',19
#     'left_wrist',20
#     'right_wrist',21

#     'jaw',22
#     'left_eye_smplhf',23
#     'right_eye_smplhf',24

#     'left_index1',
#     'left_index2',
#     'left_index3',
#     'left_middle1',
#     'left_middle2',
#     'left_middle3',
#     'left_pinky1',
#     'left_pinky2',
#     'left_pinky3',
#     'left_ring1',
#     'left_ring2',
#     'left_ring3',
#     'left_thumb1',
#     'left_thumb2',
#     'left_thumb3',
#     'right_index1',
#     'right_index2',
#     'right_index3',
#     'right_middle1',
#     'right_middle2',
#     'right_middle3',
#     'right_pinky1',
#     'right_pinky2',
#     'right_pinky3',
#     'right_ring1',
#     'right_ring2',
#     'right_ring3',
#     'right_thumb1',
#     'right_thumb2',
#     'right_thumb3',
# ]
#
# kinematic_tree = [-1,          0,          0,          0,          1,          2,          3,          4,          5,          6,          7,          8,          9,          9,          9,         12,         13,         14,         16,         17,         18,         19,         15,         15,         15,         20,         25,         26,         20,         28,         29,         20,         31,         32,         20,         34,         35,         20,         37,         38,         21,         40,         41,         21,         43,         44,         21,         46,         47,         21,         49,         50,         21,         52,         53]

# JOINT_NAMES = [
#     'pelvis',
#     'left_hip',
#     'right_hip',
#     'spine1',
#     'left_knee',
#     'right_knee',
#     'spine2',
#     'left_ankle',
#     'right_ankle',
#     'spine3',
#     'left_foot',
#     'right_foot',
#     'neck',
#     'left_collar',
#     'right_collar',
#     'head',
#     'left_shoulder',
#     'right_shoulder',
#     'left_elbow',
#     'right_elbow',
#     'left_wrist',
#     'right_wrist',
#     'jaw',
#     'left_eye_smplhf',
#     'right_eye_smplhf',
#     'left_index1',
#     'left_index2',
#     'left_index3',
#     'left_middle1',
#     'left_middle2',
#     'left_middle3',
#     'left_pinky1',
#     'left_pinky2',
#     'left_pinky3',
#     'left_ring1',
#     'left_ring2',
#     'left_ring3',
#     'left_thumb1',
#     'left_thumb2',
#     'left_thumb3',
#     'right_index1',
#     'right_index2',
#     'right_index3',
#     'right_middle1',
#     'right_middle2',
#     'right_middle3',
#     'right_pinky1',
#     'right_pinky2',
#     'right_pinky3',
#     'right_ring1',
#     'right_ring2',
#     'right_ring3',
#     'right_thumb1',
#     'right_thumb2',
#     'right_thumb3',
# ]
#

# kinematic_tree = [
#     -1, 0, 0, 0,
#     1, 2, 3, 4,
#     5, 6, 7, 8,
#     9, 9, 9, 12,
#     13, 14, 16, 17,
#     18, 19, 15, 15,
#     15, 20, 25, 26,
#     20, 28, 29, 20,
#     31, 32, 20, 34,
#     35, 20, 37, 38,
#     21, 40, 41, 21,
#     43, 44, 21, 46,
#     47, 21, 49, 50,
#     21, 52, 53
# ]

# kinematic_tree = [-1,          0,          0,          0,          1,          2,          3,          4,          5,          6,          7,          8,          9,          9,          9,         12,         13,         14,         16,         17,         18,         19,         15,         15,         15,         20,         25,         26,         20,         28,         29,         20,         31,         32,         20,         34,         35,         20,         37,         38,         21,         40,         41,         21,         43,         44,         21,         46,         47,         21,         49,         50,         21,         52,         53]

def hierarchical_softmax(x):
    """ Apply hierarchical softmax to the input tensor x.
    Args:
        x (torch.Tensor): input tensor of shape (n_point, 55+4)
    Returns:
        prob_all (torch.Tensor): output tensor of shape (n_point, 55)
    """
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    height, width, dims = x.shape
    n_point = height * width
    x = x.view(n_point, dims)
    sigmoid_x = sigmoid(x).float()

    prob_all = torch.ones(n_point, 55, device=x.device)

    # pelvis -> left_hip, right_hip, spine1
    prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid_x[:, [0]] * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid_x[:, [0]])

    # left_hip -> left_knee, right_hip -> right_knee, spine1 -> spine2
    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * sigmoid_x[:, [4, 5, 6]]
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid_x[:, [4, 5, 6]])

    # left_knee -> left_ankle, right_knee -> right_ankle, spine2 -> spine3
    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * sigmoid_x[:, [7, 8, 9]]
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid_x[:, [7, 8, 9]])

    # left_ankle -> left_foot, right_ankle -> right_foot
    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * sigmoid_x[:, [10, 11]]
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid_x[:, [10, 11]])

    # spine3 -> neck, left_collar, right_collar
    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid_x[:, [55]] * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid_x[:, [55]])

    # neck -> head
    prob_all[:, [15]] = prob_all[:, [12]] * sigmoid_x[:, [15]]
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid_x[:, [15]])

    # head -> jaw, left_eye_smplhf, right_eye_smplhf
    prob_all[:, [22, 23, 24]] = prob_all[:, [15]] * sigmoid_x[:, [56]] * softmax(x[:, [22, 23, 24]])
    prob_all[:, [15]] = prob_all[:, [15]] * (1 - sigmoid_x[:, [56]])

    # left_collar -> left_shoulder, right_collar -> right_shoulder
    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * sigmoid_x[:, [16, 17]]
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid_x[:, [16, 17]])

    # left_shoulder -> left_elbow, right_shoulder -> right_elbow
    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * sigmoid_x[:, [18, 19]]
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid_x[:, [18, 19]])

    # left_elbow -> left_wrist, right_elbow -> right_wrist
    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * sigmoid_x[:, [20, 21]]
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid_x[:, [20, 21]])

    # left_wrist -> left_index1, left_middle1, left_pinky1, left_ring1, left_thumb1
    prob_all[:, [25, 28, 31, 34, 37]] = prob_all[:, [20]] * sigmoid_x[:, [57]] * softmax(x[:, [25, 28, 31, 34, 37]])
    prob_all[:, [20]] = prob_all[:, [20]] * (1 - sigmoid_x[:, [57]])

    # right_wrist -> right_index1, right_middle1, right_pinky1, right_ring1, right_thumb1
    prob_all[:, [40, 43, 46, 49, 52]] = prob_all[:, [21]] * sigmoid_x[:, [58]] * softmax(x[:, [40, 43, 46, 49, 52]])
    prob_all[:, [21]] = prob_all[:, [21]] * (1 - sigmoid_x[:, [58]])

    # left_index1 -> left_index2, left_middle1 -> left_middle2, left_pinky1 -> left_pinky2, left_ring1 -> left_ring2, left_thumb1 -> left_thumb2,
    # right_index1 -> right_index2, right_middle1 -> right_middle2, right_pinky1 -> right_pinky2, right_ring1 -> right_ring2, right_thumb1 -> right_thumb2
    prob_all[:, [26, 29, 32, 35, 38, 41, 44, 47, 50, 53]] = prob_all[:, [25, 28, 31, 34, 37, 40, 43, 46, 49, 52]] * sigmoid_x[:, [26, 29, 32, 35, 38, 41, 44, 47, 50, 53]]
    prob_all[:, [25, 28, 31, 34, 37, 40, 43, 46, 49, 52]] = prob_all[:, [25, 28, 31, 34, 37, 40, 43, 46, 49, 52]] * (1 - sigmoid_x[:, [26, 29, 32, 35, 38, 41, 44, 47, 50, 53]])

    # left_index2 -> left_index3, left_middle2 -> left_middle3, left_pinky2 -> left_pinky3, left_ring2 -> left_ring3, left_thumb2 -> left_thumb3,
    # right_index2 -> right_index3, right_middle2 -> right_middle3, right_pinky2 -> right_pinky3, right_ring2 -> right_ring3, right_thumb2 -> right_thumb3
    prob_all[:, [27, 30, 33, 36, 39, 42, 45, 48, 51, 54]] = prob_all[:, [26, 29, 32, 35, 38, 41, 44, 47, 50, 53]] * sigmoid_x[:, [27, 30, 33, 36, 39, 42, 45, 48, 51, 54]]
    prob_all[:, [26, 29, 32, 35, 38, 41, 44, 47, 50, 53]] = prob_all[:, [26, 29, 32, 35, 38, 41, 44, 47, 50, 53]] * (1 - sigmoid_x[:, [27, 30, 33, 36, 39, 42, 45, 48, 51, 54]])

    prob_all = prob_all.reshape(height, width, 55)
    return prob_all