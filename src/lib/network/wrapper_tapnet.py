import torch


class LossWrapTapNet(torch.nn.Module):
    def __init__(self, args, model, phi, criterion, trn_embedding=None, val_embedding=None):
        self.args = args
        super(LossWrapTapNet, self).__init__()
        self.model = model
        # self.head = head
        self.phi = phi
        self.criterion = criterion
        self.trn_embedding = trn_embedding
        self.val_embedding = val_embedding

    def forward(self, input, label):
        if self.training:
            self.n_way = self.args.TRAIN.n_way
            n_support = self.args.TRAIN.n_support
            n_query = self.args.TRAIN.n_query
            n_sample = n_support + n_query
        else:
            self.n_way = self.args.TEST.n_way
            n_support = self.args.TEST.n_support
            n_query = self.args.TEST.n_query
            n_sample = n_support + n_query

        # if self.training:
        B, CXN, C, W, H = input.size()
        input = input.reshape(-1, C, W, H)
        if self.args.multi_gpus:
            input, label = input.cuda(), label.cuda()
        else:
            input, label = input.to(
                self.args.device), label.to(self.args.device)

        raw_logits = self.model(input)

        raw_logits = raw_logits.reshape(
            B, self.n_way, n_sample, -1)

        support_feats = raw_logits[:, :, :n_support]

        query_feats = raw_logits[:, :, n_support:]
        query_feats = query_feats.reshape(
            B, self.n_way*n_query, -1)
        support_mean_feats = torch.mean(
            support_feats, dim=2)

        batchsize_q = self.n_way * n_query
        for data_index in range(B):
            each_query_feats = query_feats[data_index]
            each_support_mean_feats = support_mean_feats[data_index]

            M = self.projection_space(
                average_key=each_support_mean_feats, 
                batchsize=batchsize_q,
                nb_class=self.n_way
            )

            # torch.bmm()
            r_t = torch.mm(torch.mm(each_query_feats, M), M.T)
            # r_t = F.reshape(F.batch_matmul(M,F.batch_matmul(M,query_set,transa=True)),(batchsize_q,-1))
            
            pow_t = self.compute_power(batchsize_q, each_query_feats, M, self.n_way)
            
            loss = self.compute_loss(labels[self.n_way*self.n_support:], r_t, pow_t, batchsize_q,self.nb_class_train)
        
        
        return output_all

    

    
    def compute_power(self, batchsize,key,M, nb_class, train=True,phi_ind=None):
        if train == True:
            phi_out = self.chain.l_phi.W
        else:
            phi_data = self.chain.l_phi.W.data
            phi_out = chainer.Variable(phi_data[phi_ind,:])
        phi_out_batch = F.broadcast_to(phi_out,[batchsize,nb_class, self.dimension])
        phiM = F.batch_matmul(phi_out_batch,M)
        phiMs = F.sum(phiM*phiM,axis=2)
        
        key_t = F.reshape(key,[batchsize,1,self.dimension])
        keyM = F.batch_matmul(key_t,M)
        keyMs = F.sum(keyM*keyM, axis=2)
        keyMs = F.broadcast_to(keyMs, [batchsize,nb_class])
        
        pow_t = phiMs + keyMs
        
        return pow_t
    
    
    def compute_power_avg_phi(self, batchsize, nb_class, average_key, train=False):
        avg_pow = F.sum(average_key*average_key,axis=1)
        phi = self.chain.l_phi.W
        phis = F.sum(phi*phi,axis=1)
        
        avg_pow_bd = F.broadcast_to(F.reshape(avg_pow,[len(avg_pow),1]),[len(avg_pow),len(phis)])
        wzs_bd = F.broadcast_to(F.reshape(phis,[1,len(phis)]),[len(avg_pow),len(phis)])
        
        pow_avg = avg_pow_bd + wzs_bd
        
        return pow_avg
    
    
    def compute_loss(self, t_data, r_t, pow_t, batchsize,nb_class, train=True):
        t = chainer.Variable(self.xp.array(t_data, dtype=self.xp.int32)) 
        u = 2*self.chain.l_phi(r_t)-pow_t
        return F.softmax_cross_entropy(u,t)

    def compute_accuracy(self, t_data, r_t, pow_t,batchsize, nb_class, phi_ind=None):
        ro = 2*self.chain.l_phi(r_t)
        ro_t = chainer.Variable(ro.data[:,phi_ind])
        u = ro_t-pow_t
       
        t_est = self.xp.argmax(F.softmax(u).data, axis=1)

        return (t_est == self.xp.array(t_data))
    
    def select_phi(self, average_key, avg_pow):
        u_avg = 2*self.chain.l_phi(average_key).data
        u_avg = u_avg - avg_pow.data
        u_avg_ind = cp.asnumpy(self.xp.argsort(u_avg, axis=1))
        phi_ind = np.zeros(self.nb_class_test)
        for i in range(self.nb_class_test):
            if i == 0:
                phi_ind[i] = np.int(u_avg_ind[i, self.nb_class_train-1])
            else:
                k=self.nb_class_train-1
                while u_avg_ind[i,k] in phi_ind[:i]:
                    k = k-1
                phi_ind[i] = np.int(u_avg_ind[i,k])
        return phi_ind.tolist()