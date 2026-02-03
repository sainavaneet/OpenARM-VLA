import logging
import random
import pickle

import cv2
import h5py
import os
import torch
import numpy as np
from MambaVLA.utils.sim_path import sim_framework_path
from MambaVLA.utils.embeddings.genrate_embeddings import create_embeddings_from_dataset_dir

log = logging.getLogger(__name__)


class OpenArmDataset():
    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 32,
            action_dim: int = 7,
            state_dim: int = 45,
            max_len_data: int = 60,
            chunck_size: int = 5,
            start_idx: int = 0,
            demos_per_task: int = 1,
            allowed_tasks=None,
    ):
        self.data_directory = data_directory
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_len_data = max_len_data
        self.chunck_size = chunck_size
        self.start_idx = start_idx
        self.demos_per_task = demos_per_task

        self.data_dir = sim_framework_path(self.data_directory)
        logging.info("The dataset is loading from {}".format(self.data_dir))

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.data_directory = data_directory
        self.camera_names = ["agentview", "eye_in_hand"]
        allowed_set = None
        if allowed_tasks:
            allowed_set = set(str(x) for x in allowed_tasks)

        benchmark_type = os.path.basename(data_directory)
        task_emb_dir = sim_framework_path("language_embeddings")
        task_emb_path = os.path.join(task_emb_dir, benchmark_type + ".pkl")

        if not os.path.exists(task_emb_path):
            log.warning("Task embeddings not found at %s. Generating...", task_emb_path)
            create_embeddings_from_dataset_dir(self.data_dir, output_path=task_emb_path)

        with open(task_emb_path, 'rb') as f:
            tasks = pickle.load(f)

        data_embs = []
        actions = []
        masks = []
        agentview_rgb = []
        eye_in_hand_rgb = []

        all_states = []

        file_list = os.listdir(self.data_dir)

        for file in file_list:
            if not file.endswith('.hdf5'):
                continue

            filename = os.path.basename(file).split('.')[0]
            if filename.endswith("_demo"):
                filename = filename[:-5]
            if allowed_set is not None and filename not in allowed_set:
                continue
            if filename not in tasks:
                missing_task_names = [os.path.splitext(f)[0].replace("_demo", "") for f in file_list if f.endswith(".hdf5")]
                log.warning(
                    "Task embeddings missing entries for current dataset. Regenerating embeddings at %s",
                    task_emb_path,
                )
                create_embeddings_from_dataset_dir(self.data_dir, output_path=task_emb_path)
                with open(task_emb_path, 'rb') as f:
                    tasks = pickle.load(f)
            task_emb = tasks[filename]

            f = h5py.File(os.path.join(self.data_dir, file), 'r')

            log.info("Loading demo: {}".format(file))

            demo_keys_list = list(f["data"].keys())

            indices = np.argsort([int(elem[5:]) for elem in demo_keys_list])

            for i in indices[start_idx: start_idx + demos_per_task]:

                demo_name = demo_keys_list[i]
                demo = f["data"][demo_name]
                if "num_samples" in demo.attrs:
                    demo_length = demo.attrs["num_samples"]
                else:
                    demo_length = demo["actions"].shape[0]

                zero_actions = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
                zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

                action_data = demo['actions'][:]

                zero_actions[0, :demo_length, :] = action_data
                zero_mask[0, :demo_length] = 1

                the_last_action = action_data[-1][:]

                model_view = demo['obs']['agentview_rgb'][:]
                eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]

                joint_states = demo['obs']['joint_states'][:]
                gripper_states = demo['obs']['gripper_states'][:]

                robot_states = np.concatenate((joint_states, gripper_states), axis=-1)

                actions.append(zero_actions)
                masks.append(zero_mask)

                agentview_rgb.append(model_view)
                eye_in_hand_rgb.append(eye_in_hand)

                all_states.append(robot_states)

                data_embs.append(task_emb)

            f.close()

        self.actions = torch.from_numpy(np.concatenate(actions)).float()

        self.agentview_rgb = agentview_rgb
        self.eye_in_hand_rgb = eye_in_hand_rgb

        self.all_states = all_states

        self.data_embs = data_embs
        self.tasks = tasks

        self.masks = torch.from_numpy(np.concatenate(masks)).float()

        self.num_data = len(self.agentview_rgb)

        self.slices = self.get_slices()

    def get_slices(self):
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.chunck_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.chunck_size}")
            else:
                slices += [(i, start, start + self.chunck_size) for start in range(T - self.chunck_size + 1)]

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        result = []
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.agentview_rgb[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        obs = {}

        task_emb = self.data_embs[i]

        agentview_rgb = self.agentview_rgb[i][start:start+1]
        eye_in_hand_rgb = self.eye_in_hand_rgb[i][start:start+1]

        robot_states = self.all_states[i][start:start+1]

        if isinstance(task_emb, torch.Tensor):
            task_emb = task_emb.detach().to("cpu").float()
        else:
            task_emb = torch.tensor(task_emb, dtype=torch.float32)

        agentview_rgb = torch.from_numpy(agentview_rgb).float().permute(0, 3, 1, 2) / 255.
        eye_in_hand_rgb = torch.from_numpy(eye_in_hand_rgb).float().permute(0, 3, 1, 2) / 255.

        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        obs["agentview_image"] = agentview_rgb
        obs["eye_in_hand_image"] = eye_in_hand_rgb
        obs["lang_emb"] = task_emb

        obs["robot_states"] = torch.from_numpy(robot_states).float()

        return obs, act, mask
