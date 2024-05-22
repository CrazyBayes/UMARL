from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args, get_qweight_args
import numpy as np

if __name__ == '__main__':
    for i in range(5):
        args = get_common_args()
        args = get_mixer_args(args)
        if args.alg.find('qweight_vb') > -1:
            args = get_qweight_args(args)
        if args.map == 'foraging':
            from envs.lbforaging.foraging import ForagingEnv

            field_size = 10
            players = 4
            max_food = 2
            force_coop = False
            partially_observe = True
            sight = 2
            is_print = False
            need_render = False
            seed = np.random.randint(0, 10000)
            env_id = "Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v2".format(field_size, players, max_food,
                                                                  "-coop" if force_coop else "",
                                                                  "-{}s".format(sight) if partially_observe else "")
            # print(env_id)
            # import lbforaging
            # env = gym.make(env_id)
            # print(env._get_observation_space().shape[0])
            env = ForagingEnv(field_size,
                              players,
                              max_food,
                              force_coop,
                              partially_observe,
                              sight,
                              is_print,
                              seed,
                              need_render
                              )
        elif args.map =='nstepmatrix':
            #from envs import nstep_matrix_game
            from envs_matrix.nstep_matrix_game import NStepMatrixGame
            #import  numpy as np
            steps = 10
            good_branches = 2
            env = NStepMatrixGame(
                steps,
                good_branches
            )
        else:
            env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
