from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

DATA_DIR="/mnt2/homer/datasets/mujoco_sim/franka_shoe_pick_and_place_100_20240229-121909"


class MujocoManip(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='ID of episode in file_path.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=f'{DATA_DIR}/train/*.npy'),
            'val': self._generate_examples(path=f'{DATA_DIR}/val/*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_examples(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            for k, example in enumerate(data):
                # assemble episode --> here we're assuming demos so we set reward to 1 at the end
                episode = []
                traj_len = len(example)
                for i in range(traj_len):
                    language_instruction = "put the shoe on the circle"
                    # compute Kona language embedding
                    language_embedding = self._embed([language_instruction])[0].numpy()

                    episode.append({
                        'observation': {
                            'image': example[i]["observation"]["image"].astype(np.uint8),
                            'state': example[i]["observation"]["state"].astype(np.float32),
                        },
                        'action': example[i]['action'].astype(np.float32),
                        'discount': 1.0,
                        'reward': float(i == (traj_len - 1)),
                        'is_first': i == 0,
                        'is_last': i == (traj_len - 1),
                        'is_terminal': i == (traj_len - 1),
                        'language_instruction': language_instruction,
                        'language_embedding': language_embedding.astype(np.float32),
                    })

                # create output data sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {
                        'file_path': episode_path,
                        'episode_id': k,
                    }
                }

                # if you want to skip an example for whatever reason, simply return None
                yield episode_path + str(k), sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            for id, sample in _parse_examples(sample):
                yield id, sample

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

