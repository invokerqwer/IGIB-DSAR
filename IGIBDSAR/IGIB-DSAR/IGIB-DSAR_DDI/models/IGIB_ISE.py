import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from torch_geometric.nn import Set2Set

from embedder import embedder
from layers import GINE
from utils import create_batch_mask

from torch_scatter import scatter_mean, scatter_add, scatter_std

import time

class IGIB_ISE_ModelTrainer(embedder):
    def __init__(self, args, train_df, valid_df, test_df, repeat, fold):
        embedder.__init__(self, args, train_df, valid_df, test_df, repeat, fold)

        self.model = IGIB(device = self.device, tau = self.args.tau, num_step_message_passing = self.args.message_passing,EM=self.args.EM_NUM).to(self.device)
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=self.args.patience, mode='max', verbose=True)
        
    def train(self):        
        
        loss_function_BCE = nn.BCEWithLogitsLoss(reduction='none')
        
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.train_loss = 0
            preserve = 0

            start = time.time()
            for bc, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                masks = create_batch_mask(samples)
                outputs, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)])
                loss = loss_function_BCE(outputs, samples[2].reshape(-1, 1).to(self.device).float()).mean()

                # Information Bottleneck.
                outputs, KL_Loss_1,KL_Loss_2, cont_loss_1,cont_loss_2 = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)], bottleneck = True)
                loss += loss_function_BCE(outputs, samples[2].reshape(-1, 1).to(self.device).float()).mean()
                #print(cont_loss_1)
                #print(KL_Loss_1)
                loss += self.args.beta_1 * KL_Loss_1
                loss += self.args.beta_1 * cont_loss_1
                loss += self.args.beta_2 * KL_Loss_2
                loss += self.args.beta_2 * cont_loss_2
                #print(loss)
                loss.backward()
                self.optimizer.step()
                self.train_loss += loss
                #preserve += preserve_rate

            self.epoch_time = time.time() - start

            self.model.eval()
            self.evaluate(epoch)

            self.scheduler.step(self.val_roc_score)

            # Write Statistics
            self.writer.add_scalar("stats/preservation", preserve/bc, epoch)

            # Early stopping
            if len(self.best_val_rocs) > int(self.args.es / self.args.eval_freq):
                if self.best_val_rocs[-1] == self.best_val_rocs[-int(self.args.es / self.args.eval_freq)]:
                    if self.best_val_accs[-1] == self.best_val_accs[-int(self.args.es / self.args.eval_freq)]:
                        self.is_early_stop = True
                        PATH='state_dict_model{},{},{}.pth'.format(self.args.beta_1,self.args.beta_2,self.args.EM_NUM)
                        torch.save(self.model.state_dict(),PATH)
                        
                        #model=Net()
                        #model.load_state_dict(torch.load(PATH))
                        break

        self.evaluate(epoch, final = True)
        self.writer.close()
        
        return self.best_test_roc, self.best_test_ap, self.best_test_f1, self.best_test_acc


class IGIB(nn.Module):

    def __init__(self,
                device,
                node_input_dim=133,
                edge_input_dim=14,
                node_hidden_dim=300,
                edge_hidden_dim=300,
                num_step_message_passing=3,
                tau = 1.0,
                interaction='dot',
                num_step_set2_set=2,
                num_layer_set2set=1,
                EM = 3
                ):
        super(IGIB, self).__init__()

        self.device = device
        self.tau = tau
        self.EM = EM
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction

        self.gather = GINE(self.node_input_dim, self.edge_input_dim, 
                            self.node_hidden_dim, self.num_step_message_passing,
                            )

        self.predictor = nn.Linear(8 * self.node_hidden_dim, 1)

        self.compressor = nn.Sequential(
            nn.Linear(self.node_hidden_dim*4, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1)
            )
        
        self.solvent_predictor = nn.Linear(4 * self.node_hidden_dim, 4 * self.node_hidden_dim)
        
        self.mse_loss = torch.nn.MSELoss()

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_solute = Set2Set(2 * node_hidden_dim, self.num_step_set2set)
        self.set2set_solvent = Set2Set(2 * node_hidden_dim, self.num_step_set2set)
        self.set2set_solute1 = Set2Set(2 * node_hidden_dim, 1)
        self.set2set_solvent1 = Set2Set(2 * node_hidden_dim, 1)

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def compress(self, solute_features):
        
        p = self.compressor(solute_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p
    
    def forward(self, data, bottleneck = False, test = False):

        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]

        # node embeddings after interaction phase
        solute_features = self.gather(solute)
        solvent_features = self.gather(solvent)

        # Add normalization
        self.solute_features = F.normalize(solute_features, dim = 1)
        self.solvent_features = F.normalize(solvent_features, dim = 1)
        #print(self.solute_features.size())
        #print(self.solvent_features.size())

        # Interaction phase
        # Prediction phase
        #self.solute_features = torch.cat((self.solute_features, self.solute_prime), dim=1)
        #self.solvent_features = torch.cat((self.solvent_features, self.solvent_prime), dim=1)
        # Interaction phase
        
        len_map = torch.sparse.mm(solute_len.t(), solvent_len)
        interaction_map = torch.mm(self.solute_features, self.solvent_features.t())
        ret_interaction_map = torch.clone(interaction_map)
        ret_interaction_map = interaction_map * len_map.to_dense()
        interaction_map = interaction_map * len_map.to_dense()
        self.solvent_prime = torch.mm(interaction_map.t(), self.solute_features)
        self.solute_prime = torch.mm(interaction_map, self.solvent_features)
        # Prediction phase
        self.solute_features = torch.cat((self.solute_features, self.solute_prime), dim=1)
        self.solvent_features = torch.cat((self.solvent_features, self.solvent_prime), dim=1)
        solute_0 =self.solute_features.clone()
        solvent_0 =self.solvent_features.clone()
        if test:
            pass
        # _, self.importance = self.compress(self.solute_features)

        if bottleneck:
            EM_num = self.EM
            epsilon = 1e-7

            solute_features_0 = self.solute_features.clone()
            solvent_features_0 = self.solvent_features.clone()

            for i in range(EM_num):
                # E-step
                # Pool solvent graph and update solute features
                solvent_graph_vector = scatter_mean(self.solvent_features, solvent.batch, dim=0)  # Pool solvent
                expanded_solvent_graph_vector = solvent_graph_vector[solute.batch]  # 自动按 solute.batch 对应扩展

                # 3. 拼接到 solute_features_0
                extended_solute_features = torch.cat([solute_features_0, expanded_solvent_graph_vector], dim=1)  # 拼接到最后一维
                lambda_pos, p = self.compress(extended_solute_features)
                lambda_pos = lambda_pos.reshape(-1, 1)
                lambda_neg = 1 - lambda_pos

                node_feature_mean = scatter_mean(solute_features_0, solute.batch, dim=0)[solute.batch]
                node_feature_std = scatter_std(solute_features_0, solute.batch, dim=0)[solute.batch]

                solute_noisy_node_feature_mean = lambda_pos * solute_features_0 + lambda_neg * node_feature_mean
                solute_noisy_node_feature_std = lambda_neg * node_feature_std

                solute_noisy_node_feature = solute_noisy_node_feature_mean + torch.rand_like(solute_noisy_node_feature_mean) * solute_noisy_node_feature_std

                # M-step
                # Pool solute graph and update solvent features
                solute_graph_vector = scatter_mean(solute_noisy_node_feature, solute.batch, dim=0)  # Pool solute
                expanded_solute_graph_vector = solute_graph_vector[solvent.batch]  # 自动按 solute.batch 对应扩展

                # 3. 拼接到 solute_features_0
                extended_solvent_features = torch.cat([solvent_features_0, expanded_solute_graph_vector], dim=1)  # 拼接到最后一维

                lambda_pos, p = self.compress(extended_solvent_features)
                lambda_pos = lambda_pos.reshape(-1, 1)
                lambda_neg = 1 - lambda_pos

                solvent_node_feature_mean = scatter_mean(solvent_features_0, solvent.batch, dim=0)[solvent.batch]
                solvent_node_feature_std = scatter_std(solvent_features_0, solvent.batch, dim=0)[solvent.batch]

                solvent_noisy_node_feature_mean = lambda_pos * solvent_features_0 + lambda_neg * solvent_node_feature_mean
                solvent_noisy_node_feature_std = lambda_neg * solvent_node_feature_std

                solvent_noisy_node_feature = solvent_noisy_node_feature_mean + torch.rand_like(solvent_noisy_node_feature_mean) * solvent_noisy_node_feature_std

            # Compute noisy subgraphs
            noisy_solute_subgraphs = self.set2set_solute(solute_noisy_node_feature, solute.batch)
            noisy_solvent_subgraphs = self.set2set_solvent(solvent_noisy_node_feature, solvent.batch)
            
            
            ##*************************************************************************************
            epsilon = 1e-7

            # Solute KL Loss 计算
            # 1. 计算标准差比例的平方和
            epsilon = 1e-7

            KL_tensor_1 = 0.5 * scatter_add(((solute_noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim = 1), solute.batch).reshape(-1, 1) + \
                        scatter_add((((solute_noisy_node_feature_mean - node_feature_mean)/(node_feature_std + epsilon)) ** 2), solute.batch, dim = 0)
            KL_Loss_1 = torch.mean(KL_tensor_1)

            KL_tensor_2 = 0.5 * scatter_add(((solvent_noisy_node_feature_std ** 2) / (solvent_node_feature_std + epsilon) ** 2).mean(dim = 1), solvent.batch).reshape(-1, 1) + \
                        scatter_add((((solvent_noisy_node_feature_mean - solvent_node_feature_mean)/(solvent_node_feature_std + epsilon)) ** 2), solvent.batch, dim = 0)
            KL_Loss_2 = torch.mean(KL_tensor_2)

            # Contrastive loss
            cont_loss_1 = self.contrastive_loss(noisy_solute_subgraphs, noisy_solvent_subgraphs, self.tau)
            cont_loss_2 = self.contrastive_loss(noisy_solvent_subgraphs, noisy_solute_subgraphs, self.tau)

            # Prediction
            final_features = torch.cat((noisy_solute_subgraphs, noisy_solvent_subgraphs), dim=1)
            predictions = self.predictor(final_features)

            return predictions, KL_Loss_1, KL_Loss_2, cont_loss_1, cont_loss_2

        
        else:
            self.solute_features_s2s = self.set2set_solute(self.solute_features, solute.batch)
            self.solvent_features_s2s = self.set2set_solute(self.solvent_features, solvent.batch)

            final_features = torch.cat((self.solute_features_s2s , self.solvent_features_s2s), 1)
            predictions = self.predictor(final_features)

            if test:
                return torch.sigmoid(predictions), ret_interaction_map

            else:
                return predictions, ret_interaction_map
    def get_subgraph(self, data, bottleneck = False, test = False):

        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        # node embeddings after interaction phase
        solute_features = self.gather(solute)
        solvent_features = self.gather(solvent)

        # Add normalization
        self.solute_features = F.normalize(solute_features, dim = 1)
        self.solvent_features = F.normalize(solvent_features, dim = 1)
        #print(self.solute_features.size())
        #print(self.solvent_features.size())

        # Interaction phase
        #print(len_map.size())

        # Prediction phase
        #self.solute_features = torch.cat((self.solute_features, self.solute_prime), dim=1)
        #self.solvent_features = torch.cat((self.solvent_features, self.solvent_prime), dim=1)
        # Interaction phase
        len_map = torch.sparse.mm(solute_len.t(), solvent_len)
        interaction_map = torch.mm(self.solute_features, self.solvent_features.t())
        ret_interaction_map = torch.clone(interaction_map)
        ret_interaction_map = interaction_map * len_map.to_dense()
        interaction_map = interaction_map * len_map.to_dense()
        self.solvent_prime = torch.mm(interaction_map.t(), self.solute_features)
        self.solute_prime = torch.mm(interaction_map, self.solvent_features)
        # Prediction phase
        self.solute_features = torch.cat((self.solute_features, self.solute_prime), dim=1)
        self.solvent_features = torch.cat((self.solvent_features, self.solvent_prime), dim=1)
        solute_0 =self.solute_features.clone()
        solvent_0 =self.solvent_features.clone()
        solute_sublist=[]
        solvent_sublist=[]
        if bottleneck:
            EM_num = self.EM
            for i in range(EM_num):
                if i ==0:
                    self.solvent_features=solvent_0.clone()
                else:
                    self.solvent_features=solvent_noisy_node_feature.clone()
                #Ebu 
                self.solute_features =solute_0.clone()
                interaction_map = torch.mm(self.solute_features, self.solvent_features.t())
                ret_interaction_map = torch.clone(interaction_map)
                ret_interaction_map = interaction_map * len_map.to_dense()
                interaction_map = interaction_map * len_map.to_dense()
                #solvent_prime = torch.mm(interaction_map.t(), self.solute_features)
                solute_prime = torch.mm(interaction_map, self.solvent_features)

                lambda_pos, p = self.compress(solute_prime)
                lambda_pos = lambda_pos.reshape(-1, 1)
                lambda_neg = 1 - lambda_pos
                solute_sublist.append(lambda_pos)
                # Get Stats
                preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

                static_solute_feature = self.solute_features.clone().detach()
                node_feature_mean = scatter_mean(static_solute_feature, solute.batch, dim = 0)[solute.batch]
                node_feature_std = scatter_std(static_solute_feature, solute.batch, dim = 0)[solute.batch]
                
                noisy_node_feature_mean = lambda_pos * self.solute_features + lambda_neg * node_feature_mean
                noisy_node_feature_std = lambda_neg * node_feature_std

                noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std

                #MAX
                self.solvent_features =solvent_0.clone()
                interaction_map = torch.mm(noisy_node_feature, self.solvent_features.t())
                ret_interaction_map = torch.clone(interaction_map)
                ret_interaction_map = interaction_map * len_map.to_dense()
                interaction_map = interaction_map * len_map.to_dense()
                solvent_prime = torch.mm(interaction_map.t(), noisy_node_feature)
                #solute_prime = torch.mm(interaction_map, self.solvent_features)
                lambda_pos, p = self.compress(solvent_prime)
                lambda_pos = lambda_pos.reshape(-1, 1)
                lambda_neg = 1 - lambda_pos
                solvent_sublist.append(lambda_pos)
                # Get Stats
                preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

                static_solvent_feature = self.solvent_features.clone().detach()
                solvent_node_feature_mean = scatter_mean(static_solvent_feature, solvent.batch, dim = 0)[solvent.batch]
                solvent_node_feature_std = scatter_std(static_solvent_feature, solvent.batch, dim = 0)[solvent.batch]
                
                solvent_noisy_node_feature_mean = lambda_pos * self.solvent_features + lambda_neg * solvent_node_feature_mean
                solvent_noisy_node_feature_std = lambda_neg * solvent_node_feature_std

                solvent_noisy_node_feature = solvent_noisy_node_feature_mean + torch.rand_like(solvent_noisy_node_feature_mean) * solvent_noisy_node_feature_std
        return solute_sublist,solvent_sublist
    def contrastive_loss(self, solute, solvent, tau):

        batch_size, _ = solute.size()
        solute_abs = solute.norm(dim = 1)
        solvent_abs = solvent.norm(dim = 1)        

        sim_matrix = torch.einsum('ik,jk->ij', solute, solvent) / torch.einsum('i,j->ij', solute_abs, solvent_abs)
        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss
    

    def get_checkpoints(self):
        
        return self.solute_features_s2s, self.solvent_features_s2s, self.importance