from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import Image

# Load dataset
dataset_zip = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

print('Keys in the dataset:', dataset_zip.keys())
imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
print(imgs.shape)
metadata = dataset_zip['metadata'][()]

# Define number of values per latents and functions to convert to indices
latents_sizes = metadata['latents_sizes']
print (latents_sizes)
latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))
print('latents_bases',latents_bases)
def latent_to_index(latents):
  return np.dot(latents, latents_bases).astype(int)
print ('latents_sizes:',latents_sizes.size)

def sample_latent(size=1):
  samples = np.zeros((size, latents_sizes.size))
  for lat_i, lat_size in enumerate(latents_sizes):
    samples[:, lat_i] = np.random.randint(lat_size, size=size)

  return samples

# Helper function to show images
def show_images_grid(imgs_, num_images=25):
  ncols = int(np.ceil(num_images**0.5))
  nrows = int(np.ceil(num_images / ncols))
  _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
  axes = axes.flatten()

  for ax_i, ax in enumerate(axes):
    if ax_i < num_images:
      ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
    else:
      ax.axis('off')

def show_density(imgs):
  _, ax = plt.subplots()
  ax.imshow(imgs.mean(axis=0), interpolation='nearest', cmap='Greys_r')
  ax.grid('off')
  ax.set_xticks([])
  ax.set_yticks([])


import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import tensordataset
import os

os.system('mkdir results && mkdir save_models')

torch.manual_seed(1)
batch_size = 128
log_interval = 10
epochs = 100
VAE_beta = 4.0

# Sample latents randomly
latents_train = sample_latent(size=5000)
print('latents_train',latents_train)
latents_test = sample_latent(size=1000)
print(latent_to_index(latents_train))
# Select images
imgs_train = imgs[latent_to_index(latents_train)]
imgs_test = imgs[latent_to_index(latents_test)]

train_dataset = tensordataset.TensorDataset(torch.from_numpy(imgs_train).type(torch.FloatTensor))
test_dataset = tensordataset.TensorDataset(torch.from_numpy(imgs_test).type(torch.FloatTensor))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print (train_loader)


class VAE(nn.Module):
    def __init__(self, imgSize=64*64):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(imgSize, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc21 = nn.Linear(1200, 10)
        self.fc22 = nn.Linear(1200, 10)
        self.fc3 = nn.Linear(10, 1200)
        self.fc4 = nn.Linear(1200, 1200)
        self.fc5 = nn.Linear(1200,1200)
        self.fc6 = nn.Linear(1200,imgSize)
        

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.imgSize = imgSize

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        #if self.training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
        #else:
        #return mu

    def decode(self, z):
        h3 = self.tanh(self.fc3(z))
        h4 = self.tanh(self.fc4(h3))
        h5 = self.tanh(self.fc5(h4))
        return self.sigmoid(self.fc6(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.imgSize))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=VAE_beta, imgSize=4096):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, imgSize))
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * imgSize

    return BCE + beta * KLD


model = VAE()
optimizer = optim.Adagrad(model.parameters(), lr=1e-2)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, data in enumerate(test_loader):
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar, 0.5).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n].resize(n,1,64,64),
                                  recon_batch.view(batch_size, 1, 64, 64)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, 1 + 1):
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(64, 10))
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 64, 64),
               'results/sample_' + str(epoch) + '.png')
    
    torch.save(model.state_dict(), '{0}/model_epoch_{1}.pth'.format('save_models/', epoch))

#Image('reconstruction_99.png')

Image('reconstruction_260.png')


eval_model = VAE()
#eval_model.load_state_dict(torch.load('model_epoch_2000.pth'))
eval_model.load_state_dict(torch.load('model_epoch_260.pth', map_location=lambda storage, loc: storage))


from IPython.display import Image
def fix_and_show(fix_dim=-2, fix_setting=0, latent_idx=6):
    ## Fix posX latent to left
    latents_sampled = sample_latent(size=64)
    latents_sampled[:, fix_dim] = fix_setting
    indices_sampled = latent_to_index(latents_sampled)
    imgs_sampled = imgs[indices_sampled]

    imgs_sampled = Variable(torch.from_numpy(imgs_sampled).type(torch.FloatTensor), volatile=True)

    mu, logvar = eval_model.encode(imgs_sampled.view(-1, 64*64))

    a = mu.data.numpy()
    a_arg = np.argsort(a[:, latent_idx+1])
    a = a[a_arg]
    z = eval_model.reparameterize(mu, logvar)

    samples = eval_model.decode(z)
    samples = torch.from_numpy(samples.data.numpy()[a_arg])
    save_image(samples.view(64, 1, 64, 64),'eval_samples.png')

#fix_and_show(3,0,6)

def fix_and_traverse():
    results = torch.FloatTensor(50,1,64,64).zero_()
    latent_sampled = sample_latent(size=2)
    indice_sampled = latent_to_index(latent_sampled)
    img_sampled = imgs[indice_sampled]
    show_images_grid(img_sampled,2)
    img_sampled = img_sampled[0]
    img_sampled = Variable(torch.from_numpy(img_sampled).type(torch.FloatTensor), volatile=True)
    mu, logvar = eval_model.encode(img_sampled.view(-1, 64*64))
    z = eval_model.reparameterize(mu, logvar)
    #print(logvar)
    origin_z = z.data.clone()
    for i,(fixed_dim, k) in enumerate(zip(range(10), range(40,50))):
        z = Variable(origin_z.clone())
        for mean in range(-2,3):
            #print(mean)
            #print(z)
            z_revise = eval_model.reparameterize(mu*mean, logvar)
            z[0,fixed_dim] = z_revise[0,fixed_dim]
            #print(z)
            sample = eval_model.decode(z)
            results[k] = sample.data.view(-1,64,64)
            k -= 10
    
    save_image(results,'traverse.png',nrow=10)
    print(logvar.exp())
            
        
fix_and_traverse()       
Image(filename='traverse.png')
#Image(filename='eval_samples.png')


latents_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
def curves(fixed_dim):
    lat_size = latents_sizes[fixed_dim]
    samples = np.zeros((lat_size, latents_sizes.size))
   

    for i, size in enumerate(latents_sizes):
        if i == fixed_dim:
            samples[:, i] = np.arange(0,lat_size,1)
        else: 
            random_x = np.random.randint(size)
            samples[:, i] = np.array([random_x]*lat_size)
    samples[:,1]=2
    indices_sampled = latent_to_index(samples)
    imgs_sampled = imgs[indices_sampled]
    #show_images_grid(imgs_sampled,lat_size)

    
    imgs_sampled = Variable(torch.from_numpy(imgs_sampled).type(torch.FloatTensor), volatile=True)
    mu, logvar = eval_model.encode(imgs_sampled.view(-1, 64*64))
    z = eval_model.reparameterize(mu, logvar)
    
    for i in range(10):
        plt.plot(z.data.numpy()[:,i])
    plt.ylabel('values on each dimension of z') 
    plt.xlabel(latents_names[fixed_dim]) 
    plt.show()
for i in range(1,latents_sizes.size):
    curves(i)


train_loader = torch.utils.data.DataLoader(
    newdatasets.EMNIST('../data_emnist/', split='letters', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    newdatasets.EMNIST('../data_emnist/', split='letters', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

model = VAE(28*28)

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(64, 10))
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 28, 28),
               'results_emnist/sample_' + str(epoch) + '.png')
    
    torch.save(model.state_dict(), '{0}/model_epoch_{1}.pth'.format('save_models_emnist/', epoch))
