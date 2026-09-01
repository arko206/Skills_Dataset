from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import gdown
import os
import collections
# from diffusion_agent_dataset import PushTStateDataset
from my_tf_logger.diffusion_agent_dataset import PushTStateDataset, normalize_data, unnormalize_data


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
    

### Function for performing backward transformation
def perform_backward_transformation(delta_pose_x, delta_pose_y, delta_pose_z, delta_pose_r, delta_pose_p, delta_pose_yaw, action):

    ### Converting each action array of size (1x2)to list
    list_action = action.tolist()

    print("The original action is: ", list_action)


    x = list_action[0]
    y = list_action[1]
    z = list_action[2]
    r = list_action[3]
    p = list_action[4]
    yaw = list_action[5]


    ### updating the actions of Go2 robot according to backward transformation

    translated_backward_pos_x = (x - delta_pose_x)
    translated_backward_pos_y = (y - delta_pose_y)
    translated_backward_pos_z = (z - delta_pose_z)
    translated_backward_pos_r = (r - delta_pose_r)
    translated_backward_pos_p = (p - delta_pose_p)
    translated_backward_pos_yaw = (yaw - delta_pose_yaw)


    print("The translated backward positions of Go2 are: ", translated_backward_pos_x, translated_backward_pos_y, translated_backward_pos_z, translated_backward_pos_r, translated_backward_pos_p, translated_backward_pos_yaw)


    list_action[0] = translated_backward_pos_x
    list_action[1] =  translated_backward_pos_y
    list_action[2] =  translated_backward_pos_z
    list_action[3] =  translated_backward_pos_r
    list_action[4] =  translated_backward_pos_p
    list_action[5] =  translated_backward_pos_yaw
    
    ### converting back the action list to array
    array_action = np.array(list_action)
    print("The transformed action is: ", array_action)
    return array_action

import json


user_obs_dir = "/home/arka/Demo_Report_Go2/Corr_Jump_200_epochs/Obs_user_200"
user_act_dir = "/home/arka/Demo_Report_Go2/Corr_Jump_200_epochs/Act_User_200"
# Ensure the directory exists (creates it if not)
os.makedirs(user_obs_dir, exist_ok=True)
os.makedirs(user_act_dir, exist_ok=True)






### Function for executing the backward actions
def execute_backward_actions(input_pose_x, input_pose_y, input_pose_z, input_pose_r, input_pose_p, input_pose_yaw, action_list, 
                                    delta_pose_x, delta_pose_y, delta_pose_z, delta_pose_r, delta_pose_p, delta_pose_yaw, value):
    
    print("The Length of list of actions is: ", len(action_list))

    
    ###--creating lists for storing backward observations and actions---##
    backward_observation_list = []
    backward_action_list = []

    ### resetting the environment with input values from user
    obs = np.array([input_pose_x, input_pose_y, input_pose_z, input_pose_r, input_pose_p, input_pose_yaw], dtype=np.float32)

    backward_observation_list.append(obs)
    
    print("Initial Observation in Backward environment is: ", obs)

    print("The change in positions are: ", delta_pose_x, delta_pose_y, delta_pose_z, delta_pose_r, delta_pose_p, delta_pose_yaw)

    step_index = 0
    for i in range(len(action_list)):
        for item in (action_list[i]):

            transformed_item = perform_backward_transformation(delta_pose_x, delta_pose_y, delta_pose_z, delta_pose_r, delta_pose_p, delta_pose_yaw, item)

            backward_action_list.append(transformed_item)

            print("The Executed action is: ", transformed_item)
    
            step_index += 1
            print("Present Step is:", step_index)

        ###--appending the last transformed action as the observation---###
        last_action_array = np.array(transformed_item, dtype=np.float32)
        backward_observation_list.append(last_action_array)   


    # Convert NumPy arrays to lists
    backward_action_list_as_lists = [arr.tolist() for arr in backward_action_list]

    with open(os.path.join(user_act_dir, f"User_200_Jump_diffusion_actions_{value}.json"), "w") as f:
            json.dump(backward_action_list_as_lists, f)

   

    backward_observations_as_lists = [obs.tolist() for obs in backward_observation_list]
    with open(os.path.join(user_obs_dir, f"User_200_Jump_diffusion_obs_{value}.json"), "w") as f:
        json.dump(backward_observations_as_lists, f)



















    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='/home/arka/Desktop/Go2_movement_collection/Front_Jump/Corrected_Jump_Diff_data/Corrected_Hundred_Jump_Go2.zarr')
    parser.add_argument('--pred_horizon', type=int, default=16)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--action_horizon', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    base_obs_dir = "/home/arka/Demo_Report_Go2/Diffusion_iter_200/Learning_Jump_300/Standard_Obs"
    base_act_dir = "/home/arka/Demo_Report_Go2/Diffusion_iter_200/Learning_Jump_300/Standard_Act"
    diff_act_dir = "/home/arka/Demo_Report_Go2/Diffusion_iter_200/Learning_Jump_300/Diff_200_learn_300"
    # Ensure the directory exists (creates it if not)
    os.makedirs(base_obs_dir, exist_ok=True)
    os.makedirs(base_act_dir, exist_ok=True)

    # observation and action dimensions corrsponding to pose information of Go2 robot
    obs_dim = 6
    action_dim = 6

    # parameters
    pred_horizon = args.pred_horizon
    obs_horizon = args.obs_horizon
    action_horizon = args.action_horizon
    print("pred_horizon:", pred_horizon)
    print("obs_horizon:", obs_horizon)
    print("action_horizon:", action_horizon)
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )


    # for this demo, we use DDPMScheduler with 200 diffusion iterations
    num_diffusion_iters = 200
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # device transfer
    #device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ = noise_pred_net.to(device)

    ##----(1)Loading the Evaluated Model for Testing ----#####
    load_pretrained = True
    ckpt_path = '/home/arka/Desktop/ros2_ws/Diffusion_Iteration_200/Learning_Iteration_300/Diff_(200)_train_300_Jump_Go2.ckpt'

    if load_pretrained and os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cuda')
        ema_noise_pred_net = noise_pred_net
        ema_noise_pred_net.load_state_dict(state_dict)
        print('Pretrained weights loaded.')
    else:
        print("No pretrained weights found or loading skipped.")

     # (2) create dataset from file
    dataset = PushTStateDataset(
        dataset_path=args.dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    max_steps = 100


    ##---(3) Getting the actual base poistion of Go2 robot in RViz-------###
    input_go2_base_pose = np.array([0.117942, 0.366836, 1.057089, 2.912711, -0.887423, -1.292522], dtype=np.float32)
    base_go2_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) 

    delta_x_pos, delta_y_pos, delta_z_pos = (base_go2_pose[0] - input_go2_base_pose[0]), (base_go2_pose[1] - input_go2_base_pose[1]), (base_go2_pose[2] - input_go2_base_pose[2])
    delta_r_pos, delta_p_pos, delta_yaw_pos = (base_go2_pose[3] - input_go2_base_pose[3]), (base_go2_pose[4] - input_go2_base_pose[4]), (base_go2_pose[5] - input_go2_base_pose[5])
    


   

    for iter in range(1, 101):

        ###---(3) Hardcoding the Base Pose of Go2 Robot ----###
        # base_go2_pose = np.array([0.117942, 0.366836, 1.057089, 2.912711, -0.887423, -1.292522], dtype=np.float32)
        # base_go2_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        

        print("The Observation of the Environment is: ", base_go2_pose)

        # (4) keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [base_go2_pose] * obs_horizon, maxlen=obs_horizon)
        

        print("Elements in the Observation queue are:", obs_deque)

         ###-- (3*) Creating a Bigger List for Storing the ACtions ---###
        Bigger_action_list = []

        ###----(5) Creating Action List for storing actions of the Go2 Robot----###
        fwd_action_list = []

        ###--Storing the observations of the Go2 robot----###
        obs_list = []

        done =False
        step_idx = 0
        
        ###-- (6) Evaluating the Diffusion Model --------#####
        with tqdm(total=max_steps, desc="Unitree Traj_Eval") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque)

                print("Before normalizing, the observation at step", step_idx, "is:", obs_seq)
                # (7) normalize observation
                obs_list.append(obs_seq)

                nobs = normalize_data(obs_seq, stats=stats['obs'])
                # (8) device transfer
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
                
                print("After normalizing, shape of the observation is:", nobs.shape)
                # infer action
                with torch.no_grad():
                    # (9) reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                    print("Shape of the conditioned observation is:", obs_cond.shape)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, pred_horizon, action_dim), device=device)
                    naction = noisy_action

                    print("Shape of the noisy action is:", naction.shape)

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    print("Timesteps in the noise scheduler are:", noise_scheduler.timesteps)

                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = ema_noise_pred_net(
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # (10) unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action'])

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                ###---Saving the actions in a list for future use ----###
                fwd_action_list.append(action)

                

                print("The Action taken at step ", step_idx, " is:", action)

                array_action = np.array(action, dtype=np.float32)
                print("all actions taken so far are:", array_action)


                # choose ONE 6-dim vector to represent the next observation
                #obs_deque.append(array_action[-2])    # (6,)
                obs_deque.append(array_action[-1])    # (6,)


                print("Updated Observation Queue after appending actions:", obs_deque)

                step_idx += 1
                pbar.update(1)


                for item in array_action:
                    Bigger_action_list.append(item)
                
                
                if step_idx > max_steps:
                    done = True
                if done:
                    break

        import json

        # Convert NumPy arrays to lists
        fwd_action_list_as_lists = [arr.tolist() for arr in fwd_action_list]

        with open(os.path.join(base_act_dir, f"Jump_300_diffusion_200_actions_{iter}.json"), "w") as f:
            json.dump(fwd_action_list_as_lists, f)

        observations_as_lists = [obs.tolist() for obs in obs_list]
        with open(os.path.join(base_obs_dir, f"Jump_300_diffusion_200_observations_{iter}.json"), "w") as f:
            json.dump(observations_as_lists, f)


        with open(os.path.join(diff_act_dir, f"More_300_diffusion_200_actions_{iter}.txt"), "w") as f:
            decimals = 6
            fmt = f"{{:.{decimals}f}}"
            rows = []

            for row in Bigger_action_list:
                # Format each number to 6 decimals without commas
                rows.append("[" + " ".join(fmt.format(float(v)) for v in row) + "]")

            # Write to file in the format:
            # [
            #  [ .... ]
            #  [ .... ]
            # ]
            f.write("[")
            f.write("\n ".join(rows))
            f.write("]")

        # # execute_backward_actions(
        # #     input_pose_x=input_go2_base_pose[0],
        # #     input_pose_y=input_go2_base_pose[1],
        # #     input_pose_z=input_go2_base_pose[2],
        # #     input_pose_r=input_go2_base_pose[3],
        # #     input_pose_p=input_go2_base_pose[4],
        # #     input_pose_yaw=input_go2_base_pose[5],
        # #     action_list=fwd_action_list,
        # #     delta_pose_x=delta_x_pos,
        # #     delta_pose_y=delta_y_pos,
        # #     delta_pose_z=delta_z_pos,
        # #     delta_pose_r=delta_r_pos,
        # #     delta_pose_p=delta_p_pos,
        # #     delta_pose_yaw=delta_yaw_pos,
        # #     value=iter
        # )


   


if __name__ == "__main__":
    main()  





































