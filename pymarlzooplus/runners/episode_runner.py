import datetime
from functools import partial
import numpy as np

# Import the PaperStyleRwareMentalizer class to ensure it is registered in the mentalizing registry
from pymarlzooplus.mentalizing.paper_style_rware_mentalizer import PaperStyleRwareMentalizer

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.utils.env_utils import check_env_installation


class EpisodeRunner:

    def __init__(self, args, logger):

        # Check if the requirements for the selected environment are installed
        check_env_installation(args.env, env_REGISTRY, logger)

        self.batch = None
        self.new_batch = None
        self.mac = None
        self.explorer = None
        if args.explorer == 'maven':  # MAVEN uses a noise vector which augments the observation
            self.noise = None

        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # (new) Paper style mentalizing inference settings.
        self.paper_mentalizing_enabled = getattr(
            self.args,
            "paper_mentalizing_enabled",
            False
        )

        if self.paper_mentalizing_enabled:
            self.paper_mentalizer = PaperStyleRwareMentalizer(
                output_dir=getattr(
                    self.args,
                    "paper_mentalizing_output_dir",
                    "mentalizing_results_paper_style"
                ),
                n_evidence=getattr(
                    self.args,
                    "paper_mentalizing_n_evidence",
                    50
                ),
                n_samples=getattr(
                    self.args,
                    "paper_mentalizing_n_samples",
                    100
                ),
                episode_time_limit=getattr(
                    self.args,
                    "episode_limit",
                    50
                )
            )

            print("Paper style mentalizer created.")
        else:
            self.paper_mentalizer = None

        # Initialize environment
        assert not (self.args.env == 'pettingzoo' and self.args.env_args['centralized_image_encoding'] is True), (
            "In 'episode_runner', the argument 'centralized_image_encoding' of pettingzoo should be False "
            "since there is only one environment, and thus the encoding can be considered as centralized."
        )
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        # Get info from environment to be printed
        print_info = self.env.get_print_info()
        if print_info != "None" and print_info is not None:
            # Simulate the message format of the logger defined in _logging.py
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            self.logger.console_logger.info(f"\n[INFO {current_time}] episode_runner {print_info}")

        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, explorer):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device
        )
        self.mac = mac
        self.explorer = explorer

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    # (new)
    def _get_base_env(self):
        """
        Get the underlying RWARE environment from any wrappers.
        """

        base_env = self.env

        for possible_attr in ["env", "_env", "unwrapped"]:
            try:
                possible_env = getattr(base_env, possible_attr)

                if possible_env is not None and possible_env is not base_env:
                    base_env = possible_env
            except Exception:
                pass

        return base_env

    # (new)
    def _get_requested_shelf_ids(self, base_env):
        """
        Return the Python ids of shelves currently in the request queue.
        """

        request_queue = getattr(base_env, "request_queue", [])
        return set(id(shelf) for shelf in request_queue)

    # (new)
    def _get_carried_shelf_ids(self, base_env):
        """
        Return the Python id of the shelf carried by each agent.
        If an agent is not carrying a shelf, return None for that agent.
        """

        agents = getattr(base_env, "agents", [])
        carried_shelf_ids = []

        for agent in agents:
            carrying_shelf = getattr(agent, "carrying_shelf", None)

            if carrying_shelf is None:
                carried_shelf_ids.append(None)
            else:
                carried_shelf_ids.append(id(carrying_shelf))

        return carried_shelf_ids

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        env_info = {}

        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            # (new) Include mental state input if enabled
            if getattr(self.args, "use_mental_state_input", False):
                mental_state_dim = getattr(
                    self.args,
                    "mental_state_dim",
                    6
                )  

            # (new) Get the mental state vector from the paper style mentalizer if enabled
                if self.paper_mentalizer is not None:
                    mental_state = self.paper_mentalizer.get_mental_state_vector()
                    mental_state = list(mental_state)

                    if len(mental_state) != mental_state_dim:
                        mental_state = [0.0 for _ in range(mental_state_dim)]
                    else:
                        mental_state = [float(value) for value in mental_state]
                else:
                    mental_state = [0.0 for _ in range(mental_state_dim)]
            
                mental_state_scale = float(
                    getattr(
                        self.args,
                        "mental_state_scale",
                        1.0
                    )
                )

                mental_state = [
                    float(value) * mental_state_scale
                    for value in mental_state
                ]
                pre_transition_data["mental_state"] = [mental_state]

            self.batch.update(pre_transition_data, ts=self.t)

            # In the case of MAVEN, at the beginning of each episode, sample the noise vector and add it to the batch.
            if self.args.explorer == 'maven' and self.t == 0:
                self.noise = self.explorer.sample(self.batch['state'][:, 0])
                self.batch.update({"noise": self.noise}, ts=0)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, extra_returns = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                test_mode=test_mode
            )

            ## In the case of EOI, choose actions based on the explorer.
            if self.args.explorer == 'eoi':
                actions = self.explorer.select_actions(
                    actions,
                    self.t,
                    test_mode,
                    pre_transition_data
                )

            # (new) Reward shaping for RWARE to encourage correct shelf pickups and discourage collisions.
            shaped_reward = 0.0
            correct_pickup_count = 0
            requested_shelf_ids_before = set()
            carried_shelf_ids_before = []

            if getattr(self.args, "reward_shaping_enabled", False) and not test_mode:
                base_env_before = self._get_base_env()
                requested_shelf_ids_before = self._get_requested_shelf_ids(
                    base_env_before
                )
                carried_shelf_ids_before = self._get_carried_shelf_ids(
                    base_env_before
                )

            # Step
            reward, terminated, env_info = self.env.step(actions[0])

            # (new) Reward shaping for RWARE to encourage correct shelf pickups and discourage collisions.
            if getattr(self.args, "reward_shaping_enabled", False) and not test_mode:
                base_env_after = self._get_base_env()
                carried_shelf_ids_after = self._get_carried_shelf_ids(
                    base_env_after
                )

                pickup_reward = getattr(
                    self.args,
                    "correct_pickup_reward",
                    0.1
                )

                for agent_id, carried_after in enumerate(carried_shelf_ids_after):
                    carried_before = None

                    if agent_id < len(carried_shelf_ids_before):
                        carried_before = carried_shelf_ids_before[agent_id]

                    newly_picked_up_shelf = (
                        carried_after is not None
                        and carried_after != carried_before
                    )

                    picked_requested_shelf = (
                        carried_after in requested_shelf_ids_before
                    )

                    if newly_picked_up_shelf and picked_requested_shelf:
                        shaped_reward += pickup_reward
                        correct_pickup_count += 1

            # (new) print for reward shaping
            if (getattr(self.args, "reward_shaping_debug", False)
                and shaped_reward > 0.0):
                print(
                    {
                        "debug_event": "correct_pickup_shaping",
                        "timestep": self.t_env + self.t,
                        "env_reward": reward,
                        "shaped_reward": shaped_reward,
                        "correct_pickup_count": correct_pickup_count,
                    }
                )

            # (new) Send the IQL transition into the paper style mentalizing module.
            if self.paper_mentalizer is not None:
                self.paper_mentalizer.observe_transition(
                    timestep=self.t_env + self.t,
                    observations=pre_transition_data["obs"][0],
                    actions=actions[0],
                    reward=reward,
                    terminated=terminated,
                    env_info=env_info
                )

            # Render
            if test_mode and self.args.render:
                self.env.render()

            ## Print info
            # Get info from environment to be printed
            print_info = self.env.get_print_info()
            if print_info != "None" and print_info is not None:
                # Simulate the message format of the logger defined in _logging.py
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                self.logger.console_logger.info(f"\n[INFO {current_time}] episode_runner {print_info}")

            # Keep track of episode return
            episode_return += reward

            # (new) Combine the environment reward and the shaped reward for training purposes.
            training_reward = reward + shaped_reward

            post_transition_data = {
                "actions": actions,
                "reward": [(training_reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)]
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }

        # (new) Include mental state input if enabled
        if getattr(self.args, "use_mental_state_input", False):
            mental_state_dim = getattr(
                self.args,
                "mental_state_dim",
                6
            )
            # Get the mental state vector from the paper style mentalizer if enabled
            if self.paper_mentalizer is not None:
                mental_state = self.paper_mentalizer.get_mental_state_vector()
                mental_state = list(mental_state)

                if len(mental_state) != mental_state_dim:
                    mental_state = [0.0 for _ in range(mental_state_dim)]
                else:
                    mental_state = [float(value) for value in mental_state]
            else:
                mental_state = [0.0 for _ in range(mental_state_dim)]

            mental_state_scale = float(
                getattr(
                    self.args,
                    "mental_state_scale",
                    1.0
                )
            )

            mental_state = [
                float(value) * mental_state_scale
                for value in mental_state
            ]

            last_data["mental_state"] = [mental_state]

        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, extra_returns = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # (new) Run paper style mentalizing inference at the end of the episode.
        if self.paper_mentalizer is not None:
            paper_summary = self.paper_mentalizer.finish_episode(
                episode_return=episode_return,
                episode_length=self.t
            )

            print("Paper style mentalizing summary:", paper_summary)

        return self.batch, [episode_return]

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
