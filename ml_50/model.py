import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
class hbiascorrect(nn.Module):

    def __init__(self,seq_len, seq_dim,g_embed_dim,f_embed_dim, output_dim,num_layers, num_heads, hidden_size, dropout_rate):

        super(hbiascorrect, self).__init__()

        self.seq_len = seq_len

        self.seq_dim = seq_dim 

        self.output_dim = output_dim 

        self.matlayer = nn.Linear(seq_dim, seq_dim)

        self.gembed = nn.Linear(seq_dim,g_embed_dim) 
       
        self.fembed = nn.Linear(seq_dim, f_embed_dim)

        g_encoder_layer = nn.TransformerEncoderLayer(d_model= g_embed_dim+seq_len, nhead= num_heads)

        f_encoder_layer = nn.TransformerEncoderLayer(d_model= f_embed_dim+output_dim+seq_len, nhead= num_heads)
        
        self.gtransformer = nn.TransformerEncoder(g_encoder_layer, num_layers)

        self.ftransformer = nn.TransformerEncoder(f_encoder_layer, num_layers)

        self.gpredict = nn.Sequential(nn.Linear( g_embed_dim+seq_len,output_dim),
                                    nn.ReLU())
        
        # self.fpredict = nn.Sequential(nn.Linear( f_embed_dim+output_dim+seq_len,output_dim),
        #                             nn.ReLU())
        
        self.phi = nn.Sequential(nn.Linear(f_embed_dim+output_dim+seq_len,output_dim),
                                nn.ReLU())
        
        self.rho = nn.Sequential(nn.Linear(output_dim,output_dim),
                                nn.ReLU())
        
        self.gru = nn.GRU(input_size= seq_dim+output_dim,hidden_size= hidden_size,num_layers= num_layers, dropout=dropout_rate, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_dim)

        # Add this line to enable nested tensors
        self.use_nested_tensor = True
        

    def forward(self, x):

        # print(f"x.shape: {x.shape}")

        z = self.matlayer(x)
        batch_len = x.size()[0]
        z_transposed_tensor = z.transpose(1, 2)
        zmat = torch.matmul(z, z_transposed_tensor)
        temp = 0.1
        permmat = self.pytorch_sinkhorn_iters_mask(zmat, temp, noise_factor=1.0, n_iters=20)
        
        x_wbias = torch.matmul(permmat, x)

        x_gemb = self.gembed(x)

        x_gemb_wbias = self.gembed(x_wbias)
        
        # G transformer positional encoding using identity matrix, result size:  (batch_size * seq_len * (seq_dim+seq_len))
        gtx_in = self.make_sequence_for_Gtransformer(x_gemb)
        gtx_in_wbias = self.make_sequence_for_Gtransformer(x_gemb_wbias)
        
        # print(f"gtx_in.shape: {gtx_in.shape}")
        
        gtx_out = self.gtransformer(gtx_in)
        gtx_out_wbias = self.gtransformer(gtx_in_wbias)
        
        # print(f"gtx_out.shape: {gtx_out.shape}")

        yhat = self.gpredict(gtx_out)
        yhat_wbias = self.gpredict(gtx_out_wbias)

        # print(f"yhat.shape: {yhat.shape}")
        
        # F transformer positional encoding using identity matrix, result size:  (batch_size * seq_len * (seq_dim+outpu_dim+seq_len))

        x_femb = self.fembed(x)
        x_femb_wbias = self.fembed(x)
        
        # print(f"x_femb.shape: {x_femb.shape}")

        ftx_seq = self.make_sequence_for_Ftransformer(x_femb,yhat)
        ftx_seq_wbias = self.make_sequence_for_Ftransformer(x_femb_wbias , yhat_wbias)

        # print(f"ftx_seq.shape: {ftx_seq.shape}")
        
        #append sequence[xi,yi,1,1,1,..1] batch_size*seq_len*feature_size -> (batch_size*seq_len)*(seq_len+1)*feature_size
        
        ftx_in = self.ftransformer_processing(ftx_seq,yhat) 
        ftx_in_wbias = self.ftransformer_processing(ftx_seq_wbias,yhat_wbias)

        # print(f"ftx in shape: {ftx_in.shape}")
        
        ftx_out = self.ftransformer(ftx_in)
        ftx_out_wbias = self.ftransformer(ftx_in_wbias)

        # ftx_out= torch.sum(ftx_out,dim=1)
        # ftx_out_wbias= torch.sum(ftx_out_wbias,dim=1)

        # print(f"ftx_out.shape: {ftx_out.shape}")

        # Use the latent code corresponding to x_i for each i in seq_len
        
        phi_in= ftx_out[:,self.seq_len,:]
        phi_in_wbias= ftx_out_wbias[:,self.seq_len,:]

        # print(f"phi_in.shape: {phi_in.shape}")

        # print(f"ftx out shape: {ftx_out.shape}")

        # fpred_out= self.fpredict(ftx_out)
        # fpred_out_wbias= self.fpredict(ftx_out_wbias)

        # # print(f"fpred_out.shape: {fpred_out.shape}")

        phi_out = self.phi(phi_in)
        phi_out_wbias = self.phi(phi_in_wbias)

        # print(f"phi_in.shape: {phi_in.shape}")
        # print(f"phi_out.shape: {phi_out.shape}")
        # input(" ")

        rho_seq= torch.reshape(phi_out,(batch_len, self.seq_len, self.output_dim))
        rho_seq_wbias= torch.reshape(phi_out_wbias,(batch_len, self.seq_len, self.output_dim))

        # print(f"rho_seq.shape: {rho_seq.shape}")
        
        # rho_in = torch.sum(rho_seq,dim=1)
        # rho_in_wbias = torch.sum(rho_seq_wbias,dim=1)

        # rho_out = self.rho(rho_in)
        # rho_out_wbias = self.rho(rho_in_wbias)

        # print(f"rho_in.shape: {rho_in.shape}")

        rho_out = torch.sum(rho_seq,dim=1)
        rho_out_wbias = torch.sum(rho_seq_wbias,dim=1)

        
        # print(f"rho_out: {rho_out.shape}")

        rho_repeat = rho_out.repeat_interleave(self.seq_len).reshape((-1,self.seq_len,self.output_dim))
        rho_repeat_wbias= rho_out_wbias.repeat_interleave(self.seq_len).reshape((-1,self.seq_len,self.output_dim))

        # print(f"rho_repeat: {rho_repeat.shape}")

        gru_in = torch.cat((x,rho_repeat),dim=-1)
        gru_in_wbias = torch.cat((x,rho_repeat_wbias),dim=-1)

        # print(f"gru_in.shape: {gru_in.shape}")

        gru_out,_ = self.gru(gru_in) 
        gru_out_wbias,_ = self.gru(gru_in_wbias)

        yhat_out = self.fc(gru_out)
        yhat_out_wbias = self.fc(gru_out_wbias)

        # print(f"yhat1.shape: {yhat1.shape}")

        return yhat_out, yhat_out_wbias, yhat

    def ftransformer_processing(self,input_seq,output):
        device= input_seq
        seq_len = input_seq.size()[1]
        batch_size = input_seq.size()[0]
       
        
        new_sequence = []
        for inps in input_seq:   
            for entry in inps:
                z = entry.clone()  
                z[ seq_len+1 : ] = 1  
                z = z.unsqueeze(dim=0)
                z = torch.cat((inps, z), dim=-2)
                new_sequence.append(z)
        new_sequence_tensor = torch.stack(new_sequence)
        return new_sequence_tensor
       
    
    def make_sequence_for_Gtransformer(self, input):
        device= input.device
        seq_len = input.size()[1]
        batch_size = input.size()[0]
        identity_matrix = torch.eye(seq_len).unsqueeze(0).expand(batch_size, -1, -1).to(device)
        sequence = torch.cat((input,identity_matrix),dim=-1)
        return sequence


    def make_sequence_for_Ftransformer(self, input ,output):
        device= input.device
        seq_len = input.size()[1]
        batch_size = input.size()[0]
        identity_matrix = torch.eye(seq_len).unsqueeze(0).expand(batch_size, -1, -1).to(device)
        sequence = torch.cat((input,output , identity_matrix),dim=-1)
        # # print('sequence.shape')
        # # print(sequence.shape)
        # # print("ok")
        return sequence


    def sample_gumbel(self, shape, device):
        U = torch.rand(shape).float().to(device)
        
        self.eps = 1
        return -torch.log(self.eps - torch.log(U + self.eps))

    def pytorch_sinkhorn_iters_mask(self, log_alpha, temp, noise_factor=1.0, n_iters=20):
        device = log_alpha.device  # Get the device of the input tensor
        batch, nt = log_alpha.size(0), log_alpha.size(1)
        log_alpha = log_alpha.view(-1, nt, nt)
        noise = self.sample_gumbel((batch, nt, nt), device=device) * noise_factor

        log_alpha = log_alpha + noise
        log_alpha = log_alpha / temp

        for _ in range(n_iters):
            log_alpha_copy = log_alpha.clone()  # Create a copy of log_alpha
            log_alpha_copy -= torch.logsumexp(log_alpha, dim=2, keepdim=True)
            z = log_alpha_copy.clone()
            log_alpha_copy -= torch.logsumexp(z, dim=1, keepdim=True)
            log_alpha = log_alpha_copy  # Assign the updated value back to log_alpha

        return torch.exp(log_alpha)


class Permscore(nn.Module):
  def __init__(self, input_size, hidden_size, output_dim):
        super(Permscore, self).__init__()
        
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, 1)
        

  def forward(self, x):
        
        out = self.fc(x)
        
        return out



class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out
