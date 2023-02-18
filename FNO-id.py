import torch
import json
import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
torch.manual_seed(0)
np.random.seed(0)
cosine= torch.nn.CosineSimilarity(dim=1)

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width*2)  # output channel_dim is 1: u1(x)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

################################################################
#  configurations
################################################################

def relationship(dict_relation_emb ,filename_relation):
  with open(filename_relation) as f:
      for line in f:
          words = line.strip().split("\t")
          relation = words[1]
          if relation in dict_relation_emb:
            continue
          else:
            dict_relation_emb[relation] = model.encode(relation)
      return dict_relation_emb

def node(file_name):
    dict_node_emb={}
    f = open(file_name)
    data = json.load(f)
    for element in data:
      description = data[element]["description"]
      if description == "None" or description == [] or description == None:
        continue
      dict_node_emb[element] = model.encode(description)
    return dict_node_emb


def preprocess( dict_node_emb, filename):
  # words[0]: head
  # words[1]: relation
  # words[2]: tail
  
  edges = {}
  count = {}
  with open(filename) as f:
      for line in f:
          words = line.strip().split("\t")
          if words[0] in dict_node_emb and words[2] in dict_node_emb:
            
            if words[1] in edges:
                edges[words[1]][0].append(words[0])
                edges[words[1]][1].append(words[2])
                count[words[1]] += 1
            else:
                edges[words[1]] = [[words[0]],[words[2]]]
                count[words[1]] = 0
  return count, edges


############ dictiony of relations' embedding ###########################
d0={}
d1 = relationship(d0, "/content/datasets_knowledge_embedding/FB15k-237/train.txt")
dict_relation_emb = relationship(d1, "/content/datasets_knowledge_embedding/FB15k-237/test.txt")

############ dictiony of nodes' embedding ###############################
dict_node_emb = node('/content/datasets_knowledge_embedding/FB15k-237/entity2wikidata.json')

############ preprocess data for test and train #########################
count_train,  train_pos_edges = preprocess(dict_node_emb, "/content/datasets_knowledge_embedding/FB15k-237/train.txt")
count_test,  test_pos_edges = preprocess( dict_node_emb, "/content/datasets_knowledge_embedding/FB15k-237/test.txt")



def train(count_train, count_test, train_pos_edges, test_pos_edges):
    batch_size = 10
    ntrain = count_train - count_train % batch_size
    count_train = ntrain
    ntest = count_test -count_test % batch_size
    print("___>", ntrain, ntest)
    count_test = ntest
    dim =384
    sub = 2 #subsampling rate
    h = dim * 2 // sub #total grid size divided by the subsampling rate
    s = h

    learning_rate = 0.001
    epochs = 50
    iterations = epochs*(ntrain//batch_size)

    modes = 16
    width = 64

    ###################################################
    # read data
    ###################################################

    # Data is of the shape (number of samples, grid size)
    #dataloader = MatReader('data/burgers_data_R10.mat')
    #x_data = dataloader.read_field('a')[:,::sub]
    #y_data = dataloader.read_field('u')[:,::sub]

    # x_train = x_data[:ntrain,:]
    # print(type(x_train))
    # y_train = y_data[:ntrain,:]
    # print(type(y_train))
    # x_test = x_data[-ntest:,:]
    # y_test = y_data[-ntest:,:]

    ##################################

    x_train = torch.rand(count_train, dim)
    y_train = torch.rand(count_train, dim)
    x_test = torch.rand(count_test, dim)
    y_test = torch.rand(count_test, dim)

    for i in range (count_train):

        #print(train_pos_edges[0])
        head = train_pos_edges[0][i]
        #print("--->", element)
        x_train[i] = torch.from_numpy(dict_node_emb[head])

    for i in range (count_train):
        tail = train_pos_edges[1][i]
        y_train[i] = torch.from_numpy(dict_node_emb[tail])

    for i in range (count_test):
        head = test_pos_edges[0][i]
        x_test[i] = torch.from_numpy(dict_node_emb[head])

    for i in range (count_test):
        tail = test_pos_edges[1][i]
        y_test[i] = torch.from_numpy(dict_node_emb[tail])


    x_train = x_train.reshape(ntrain,s,1)
    x_test = x_test.reshape(ntest,s,1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    x_train =  x_train.to(torch.device("cuda:0"))
    y_train =  y_train.to(torch.device("cuda:0"))
    x_test =  x_test.to(torch.device("cuda:0"))
    y_test =  y_test.to(torch.device("cuda:0"))   
    # model
    model = FNO1d(modes, width).cuda()
    #print(count_params(model))

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)
            #print("x",  x.shape)
            #print("y", y.shape)
            #print("out",  out.shape)
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            optimizer.step()
            scheduler.step()
            train_mse += mse.item()
            train_l2 += l2.item()
            train_mse /= len(train_loader)
            train_l2 /= ntrain
            #test_l2 /= ntest
            t2 = default_timer()
            #print(ep, t2-t1, train_mse, train_l2)
            model.eval()
        #test_l2 = 0.0
    with torch.no_grad():
        index_y = 0 
        rank_list=[]
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            
            #print(len(y))
            out = model(x)
            
            for i in range(len(out)):
              
              
              
              #print(y_test.data.shape)
              #rint(y_train.data.shape)
              #print(out[i].get_device())
              whole =  cosine(out[i].view(-1, dim), y_test.view(-1, dim))
              whole_sorted = torch.sort(whole)[0]
              index = torch.where(torch.sort(whole)[1] == index_y) 
              rank = 1/(index[0].item() + 1)
              #it =  cosine(out[i].view(-1, dim), y[i].view(-1, dim))
              #print(it.shape)
              #print("it", it)
              
              #print("index:::", index[0].item())
              rank_list.append(rank)
              index_y += 1 
            print("MRR", sum(rank_list)/len(rank_list))
                
                #test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()



    # torch.save(model, 'model/ns_fourier_burgers')
    #pred = torch.zeros(y_test.shape)
    #index = 0
    #test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    #with torch.no_grad():
    #    for x, y in test_loader:
    #        test_l2 = 0
    #        x, y = x.cuda(), y.cuda()

    #       out = model(x).view(-1)
    #        pred[index] = out

    #       test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
    #       print(index, test_l2)
    #       index = index + 1

    # scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})

    
    

#print("test_pos_edges", test_pos_edges)
for key in test_pos_edges:
  if key in train_pos_edges:
    print("RESULT FOR", key)
    train(count_train[key], count_test[key], train_pos_edges[key], test_pos_edges[key])

