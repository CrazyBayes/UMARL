import torch.nn as nn
import torch
import torch.nn.functional as F
from network.qmix_net import QMixNet

class conv_Pro(nn.Module):
    def __init__(self, args):
        super(conv_Pro, self).__init__()
        self.fc_1 = nn.Linear(args.input_shape_pro_shape + 1 + args.n_agents, args.state_Pro_dim)
        self.fc_2 = nn.Linear(args.state_Pro_dim, args.noise_dim)

    def forward(self, conv, q_value, agents_ids):
        q_value = q_value.reshape(q_value.shape[0], q_value.shape[1], q_value.shape[2], 1)
        input = torch.cat([conv, q_value, agents_ids], dim=-1)
        x = F.relu(self.fc_1(input))
        q = self.fc_2(x)
        prob = F.softmax(q, dim=-1)
        return prob

class conv_stat_Pro(nn.Module):
    def __init__(self, args):
        super(conv_stat_Pro, self).__init__()
        self.fc_1 = nn.Linear(args.state_shape + args.input_shape_pro_shape + 1 + args.n_agents, args.state_Pro_dim)
        self.fc_2 = nn.Linear(args.state_Pro_dim, args.noise_dim)
        self.args = args

    def forward(self, state, conv, q_value, agents_ids):
        state = state.unsqueeze(2)
        state = state.repeat(1, 1, self.args.n_agents, 1)
        q_value = q_value.reshape(q_value.shape[0], q_value.shape[1], q_value.shape[2], 1)
        input = torch.cat([state,conv, q_value, agents_ids], dim=-1)
        x = F.relu(self.fc_1(input))
        q = self.fc_2(x)
        prob = F.softmax(q, dim=-1)
        return prob


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, concat=False):
        super(GraphAttentionLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.concat = concat

        self.FC_W = nn.Linear(in_features,out_features, bias=False)
        self.FC_a = nn.Linear(2*out_features,1, bias=False)

        #self.W = nn.Parameter(torch.zeros(size=(in_features,out_features))).cuda()
        #nn.init.xavier_uniform(self.W.data, gain = 1.414)
        #self.a = nn.Parameter(torch.zeros(size=(2*out_features,1))).cuda()
        #nn.init.xavier_uniform(self.a.data, gain = 1.414)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input, adj,adj_we):
        shape = input.shape

        #h = torch.matmul(input, self.W)
        h = self.FC_W(input)
        N = adj.shape[0]



        a_input = torch.cat([h.repeat(1,1,1,N).view(shape[0], shape[1], N*N, -1),
                             h.repeat(1,1,N,1)],dim=3).view(shape[0], shape[1],
                                                            N, -1, 2*self.out_features)
        #e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(4))
        e = self.leakyrelu(self.FC_a(a_input).squeeze(4))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        attention = attention * adj_we
        attention = F.softmax(attention, dim=3)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return nn.functional.elu(h_prime)
        else:
            return h_prime

class GAT_multihead_conv(nn.Module):
    def __init__(self, in_features, out_features, nheads):
        super(GAT_multihead_conv, self).__init__()

        self.attentions = [GraphAttentionLayer(in_features,out_features)
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)

    def forward(self, input, adj, adj_we):
        x = torch.stack([att(input,adj, adj_we) for att in self.attentions]).mean(dim=0)


        return x

class GraphConvo(nn.Module):
    def __init__(self, args):
        super(GraphConvo,self).__init__()
        #from torch_geometric.nn import GATConv
        ##这里将卷积的次数设置为3
        self.heads = args.convo_heads
        self.hidden_dim = args.hidden_dim
        self.in_features = args.obs_shape ##必须和Agent的obs_shape一致
        self.out_features = args.out_features
        self.con_num = args.con_num

        self.gat1 = GAT_multihead_conv(self.in_features, self.hidden_dim,  self.heads)
        self.gat2 = GAT_multihead_conv(self.hidden_dim, self.hidden_dim,  self.heads)
        self.gat3 = GAT_multihead_conv(self.hidden_dim, self.hidden_dim,  self.heads)

        #self.gat1 = GATConv(self.in_features, self.hidden_dim, heads = self.heads, concat=False)
        #self.gat2 = GATConv(self.hidden_dim,self.hidden_dim, heads = self.heads, concat=False)
        #self.gat3 = GATConv(self.hidden_dim, self.out_features, heads = self.heads, concat=False)
    def forward(self, input, adj, adj_we):
        convo_1 = self.gat1(input,adj, adj_we)
        convo_1 = nn.functional.relu(convo_1)
        convo_2 = self.gat2(convo_1, adj, adj_we)
        convo_2 = nn.functional.relu(convo_2)
        convo_3 = self.gat3(convo_2,adj, adj_we)
        convo_result = torch.cat([convo_1,convo_2,convo_3], dim=-1)
        return  convo_result

class QWeightNet_weighting(nn.Module):
    def __init__(self, args):
        super(QWeightNet_weighting, self).__init__()
        self.args = args
        self.input_size_state = args.state_shape
        self.input_size_obs = args.weight_input_size * args.con_num
        self.attend_dim = args.weight_attend_dim
        self.attend_heads = args.weight_atten_heads

        self.key_extractors = nn.ModuleList()
        self.query_extractors = nn.ModuleList()
        for i in range(self.attend_heads):
            self.key_extractors.append(nn.Linear(self.input_size_state, self.attend_dim, bias=False))
            self.query_extractors.append(nn.Linear(self.input_size_obs, self.attend_dim, bias=False))

    def forward(self, state, obs_gat, Q_values):
        obs_gat = obs_gat.view(obs_gat.shape[2], obs_gat.shape[0], obs_gat.shape[1], obs_gat.shape[3])
        state_key = [key_extractors(state) for key_extractors in self.key_extractors]
        obs_gat_query = [[query_extractor(obs_gat_i) for obs_gat_i in obs_gat]
                         for query_extractor in self.query_extractors]
        W_i_1 = []

        for i, state_key_hi in enumerate(state_key):
            W_i_1_eh = []
            key_tr = state_key_hi.view(state_key_hi.shape[0], state_key_hi.shape[1],-1, 1)
            for obs_gat_query_hi in obs_gat_query[i]:
                obs_gat_query_tr = obs_gat_query_hi.view(obs_gat_query_hi.shape[0],
                                                      obs_gat_query_hi.shape[1],1,-1)
                W_i_1_eh.append(torch.matmul(obs_gat_query_tr,key_tr).view(state_key_hi.shape[0], state_key_hi.shape[1],-1))
            W_i_1_eh = torch.stack(W_i_1_eh, dim=2)
            W_i_1.append(W_i_1_eh)

        W_i_1 = torch.stack(W_i_1, dim=0)
        W_i_1 = W_i_1.mean(dim=0)
        W_i = torch.nn.functional.softmax(W_i_1,dim=2)
        W_i_1 = W_i_1.reshape(W_i.shape)
        W_i = W_i.reshape(Q_values.shape)
        Q = torch.mul(Q_values, W_i)
        Q = Q.sum(axis=2)

        return Q, W_i_1

class QMixNet_individual(nn.Module):
    def __init__(self, args):
        super(QMixNet_individual, self).__init__()
        self.args = args
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        if args.two_hyper_layers: ###state_shape 要换成convnet的output
            self.hyper_w1 = nn.Sequential(nn.Linear(args.qmix_input_shape + args.n_agents,
                                                    args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(args.qmix_input_shape + args.n_agents
                                                    , args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.qmix_input_shape+ args.n_agents,  args.qmix_hidden_dim)
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Linear(args.qmix_input_shape + args.n_agents, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.qmix_input_shape + args.n_agents, args.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 =nn.Sequential(nn.Linear(args.qmix_input_shape + args.n_agents,
                                               args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        episode_num = q_values.size(0)
        q_values = q_values.reshape(q_values.shape[0], q_values.shape[1],
                                    q_values.shape[2], 1)

        q_values = q_values.view(-1,self.args.n_agents, 1, 1)  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
        states = states.reshape(-1,self.args.n_agents, 1, self.args.qmix_input_shape+self.args.n_agents)  # (episode_num * max_episode_len, state_shape)
        w1 = torch.abs(self.hyper_w1(states))  # (1920, 160)
        b1 = self.hyper_b1(states)  # (1920, 32)
        w1 = w1.view(-1, self.args.n_agents, 1, self.args.qmix_hidden_dim,)  # (1920, 5, 32)
        b1 = b1.view(-1, self.args.n_agents, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)
        hidden = F.elu(torch.matmul(q_values, w1) + b1)  # (1920, 1, 32)

        w2 = torch.abs(self.hyper_w2(states))  # (1920, 32)
        b2 = self.hyper_b2(states)  # (1920, 1)
        w2 = w2.view(-1, self.args.n_agents,self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1,self.args.n_agents, 1, 1)  # (1920, 1， 1)

        q_total = torch.matmul(hidden, w2) + b2  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1,self.args.n_agents)
        #q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)
        return q_total

class Qweight(nn.Module):
    def __init__(self, args):
        super(Qweight, self).__init__()
        self.args = args
        self.each_agent_critic_input_shape = args.state_shape+args.input_shape_critic + args.n_actions + args.n_agents
        #self.each_agent_critic_input_shape = args.input_shape_critic + args.n_actions + args.n_agents
        self.graph_con_net = GraphConvo(args)
        self.qweight_net = QMixNet_individual(args) ##Weighting_network: Attention weighting
        #self.qweight_net = QMixNet(args) #QMIX:Mixing network
    def forward(self, stat, obs, q_values, agent_ids, matrix, weighmatrix):
        conv = self.graph_con_net(obs, matrix, weighmatrix)
        #stat_ex = stat
        #stat_ex = stat_ex.unsqueeze(2)
        #stat_ex = stat_ex.repeat(1,1,self.args.n_agents,1)
        #if self.args.reuse_network:
        #    inputs = torch.cat([stat_ex,conv, actions, agent_ids], dim=-1)
            #inputs = torch.cat([conv, actions, agent_ids], dim=-1)
        #else:
        #    inputs = torch.cat([conv, actions], dim=-1)
        #total_q = self.qweight_net(q_values, stat) #Attentional weighting Mixing Network
        one_hotAgent = torch.eye(self.args.n_agents)
        one_hotAgent = one_hotAgent.expand((conv.shape[0],conv.shape[1],
                                            one_hotAgent.shape[0], one_hotAgent.shape[1]))
        if self.args.cuda:
            one_hotAgent = one_hotAgent.cuda()
        conv_cat = torch.cat([conv, one_hotAgent], dim=-1)
        total_q = self.qweight_net(q_values, conv_cat) #
        total_q =  total_q.sum(dim=-1)
        total_q = total_q.view(total_q.shape[0], total_q.shape[1], 1)
        return total_q, conv



