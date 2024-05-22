from runner import Runner
#from smac.env import StarCraft2Env
from envs.grf.academy_3_vs_1_with_keeper import Academy_3_vs_1_with_Keeper
from envs.grf.academy_corner import Academy_Corner
from envs.grf.academy_pass_and_shoot_with_keeper import Academy_Pass_and_Shoot_with_Keeper
from envs.grf.academy_counterattack_hard import Academy_Counterattack_Hard
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args, get_qweight_args


if __name__ == '__main__':
    for i in range(5):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        if args.alg.find('qweight_vb') > -1:
            args = get_qweight_args(args)
        if args.map == "academy_3_vs_1_with_keeper":
            dense_reward = False
            write_full_episode_dumps =  False
            write_goal_dumps =  False
            dump_freq = 1000
            render = False
            n_agents = 3
            time_limit = 150
            time_step = 0
            obs_dim = 26
            env_name = "academy_3_vs_1_with_keeper"
            stacked = False
            representation = "simple115v2"
            rewards = "scoring,checkpoints"#"scoring"
            logdir = "football_dumps"
            write_video = False
            number_of_right_players_agent_controls = 0
            seed = 0
            reward_sparse = False
            reward_max = 10
            reward_positive = False
            reward_reset_punish = True
            env = Academy_3_vs_1_with_Keeper(dense_reward=dense_reward,
                                write_full_episode_dumps=write_full_episode_dumps,
                                write_goal_dumps=write_goal_dumps,
                                dump_freq=dump_freq,
                                render=render,
                                n_agents=n_agents,
                                time_limit=time_limit,
                                time_step=time_step,
                                obs_dim=obs_dim,
                                env_name=env_name,
                                stacked=stacked,
                                representation=representation,
                                rewards=rewards,
                                logdir=logdir,
                                write_video=write_video,
                                number_of_right_players_agent_controls=number_of_right_players_agent_controls,
                                reward_sparse=reward_sparse,
                                reward_max=reward_max,
                                reward_positive=reward_positive,
                                reward_reset_punish=reward_reset_punish,
                                seed=seed)
        elif args.map == "academy_corner":
            dense_reward = False
            write_full_episode_dumps = False
            write_goal_dumps = False
            dump_freq = 1000
            render = False
            n_agents = 4
            time_limit = 150
            time_step = 0
            obs_dim = 34
            env_name = "academy_corner"
            stacked = False
            representation = "simple115v2"
            rewards = "scoring,checkpoints"
            logdir = "football_dumps"
            write_video = False
            number_of_right_players_agent_controls =0
            seed = 0
            reward_sparse = False
            reward_max = 10
            reward_positive = False
            reward_reset_punish = True
            env = Academy_Corner(dense_reward=dense_reward,
                                             write_full_episode_dumps=write_full_episode_dumps,
                                             write_goal_dumps=write_goal_dumps,
                                             dump_freq=dump_freq,
                                             render=render,
                                             n_agents=n_agents,
                                             time_limit=time_limit,
                                             time_step=time_step,
                                             obs_dim=obs_dim,
                                             env_name=env_name,
                                             stacked=stacked,
                                             representation=representation,
                                             rewards=rewards,
                                             logdir=logdir,
                                             write_video=write_video,
                                             number_of_right_players_agent_controls=number_of_right_players_agent_controls,
                                             reward_sparse=reward_sparse,
                                             reward_max=reward_max,
                                             reward_positive=reward_positive,
                                             reward_reset_punish=reward_reset_punish,
                                             seed=seed)

        elif args.map == "academy_pass_and_shoot_with_keeper":
            dense_reward = False
            write_full_episode_dumps = False
            write_goal_dumps = False
            dump_freq = 1000
            render = False
            n_agents = 2
            time_limit = 150
            time_step = 0
            obs_dim = 22
            env_name = "academy_pass_and_shoot_with_keeper"
            stacked = False
            representation = "simple115v2"
            rewards = "scoring,checkpoints"
            logdir = "football_dumps"
            write_video = False
            number_of_right_players_agent_controls = 0
            seed = 0
            reward_sparse = False
            reward_max = 10
            reward_positive = False
            reward_reset_punish = True
            env = Academy_Pass_and_Shoot_with_Keeper(dense_reward=dense_reward,
                                             write_full_episode_dumps=write_full_episode_dumps,
                                             write_goal_dumps=write_goal_dumps,
                                             dump_freq=dump_freq,
                                             render=render,
                                             n_agents=n_agents,
                                             time_limit=time_limit,
                                             time_step=time_step,
                                             obs_dim=obs_dim,
                                             env_name=env_name,
                                             stacked=stacked,
                                             representation=representation,
                                             rewards=rewards,
                                             logdir=logdir,
                                             write_video=write_video,
                                             number_of_right_players_agent_controls=number_of_right_players_agent_controls,
                                             reward_sparse=reward_sparse,
                                             reward_max=reward_max,
                                             reward_positive=reward_positive,
                                             reward_reset_punish=reward_reset_punish,
                                             seed=seed)
        elif args.map == "academy_counterattack_hard":
            dense_reward = False
            write_full_episode_dumps = False
            write_goal_dumps = False
            dump_freq = 1000
            render = False
            n_agents = 4
            time_limit = 150
            time_step = 0
            obs_dim = 34
            env_name = "academy_counterattack_hard"
            stacked = False
            representation = "simple115v2"
            rewards = "scoring,checkpoints"
            logdir = "football_dumps"
            write_video = False
            number_of_right_players_agent_controls = 0
            seed = 0
            reward_sparse = False
            reward_max = 10
            reward_positive= False
            reward_reset_punish= True
            env = Academy_Counterattack_Hard(dense_reward=dense_reward,
                                             write_full_episode_dumps=write_full_episode_dumps,
                                             write_goal_dumps=write_goal_dumps,
                                             dump_freq=dump_freq,
                                             render=render,
                                             n_agents=n_agents,
                                             time_limit=time_limit,
                                             time_step=time_step,
                                             obs_dim=obs_dim,
                                             env_name=env_name,
                                             stacked=stacked,
                                             representation=representation,
                                             rewards=rewards,
                                             logdir=logdir,
                                             write_video=write_video,
                                             number_of_right_players_agent_controls=number_of_right_players_agent_controls,
                                             reward_sparse=reward_sparse,
                                             reward_max=reward_max,
                                             reward_positive=reward_positive,
                                             reward_reset_punish=reward_reset_punish,
                                             seed=seed)
        print("环境初始化完成")
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
