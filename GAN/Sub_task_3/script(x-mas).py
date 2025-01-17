'''
Made By Rudra in Spyder, 

Used Pytorch as framework and 2 simple linear layers in
discriminator and generator instead of CNN because CNN was not 
giving good results

Don't know why but succesfully wasted enough time in learning 
and implementing CNN (crying inside)

Implemented model on MNIST dataset beacuse it was blank and white, hehehee
Otherwise my pc would blast ..... and it was easy to plot lol

Then trained my model using mathsssssss(everywhere is maths, uffff)

And at end of each epoch, I generated image from a random noise which is fixed for all epochs

So Enjoyyyyyy 

Copyright @ Great Rudra 2025
'''
# %%
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    def __init__(self, input):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(input, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256*8),
            nn.LeakyReLU(0.01),
            nn.Linear(256*8, img_dim),
            nn.Tanh(),  
        )

    def forward(self, x):
        return self.gen(x)

# %%

transforms_dataset = transforms.Compose(
    [   
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset_tree = datasets.ImageFolder(root=r"D:\Induction_Task\Data_2\undecorated",transform=transforms_dataset)
dataset_xmas = datasets.ImageFolder(root=r"D:\Induction_Task\Data_2\decorated",transform=transforms_dataset)

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.0006
z_dim = 64*8
image_dim = 128 * 128 * 3 
batch_size_tree = len(dataset_tree)
batch_size_xmas = 64
num_epochs = 1000
check_number = 1
# %%


disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((check_number, z_dim)).to(device)


# loader_tree = DataLoader(dataset_tree, batch_size=batch_size_tree, shuffle=True)
loader_xmas = DataLoader(dataset_xmas, batch_size=batch_size_xmas, shuffle=True)

opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)
opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader_xmas):
        real = real.view(-1, image_dim).to(device)

        # Optimizing our lovely detective :)
        noise = torch.randn(batch_size_xmas, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()
        
        # Now Turn Of Generator
        output = disc(fake)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        
        if batch_idx == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )
    
            with torch.no_grad():
                gen_image = gen(fixed_noise)
                print("\nDiscriminator's prediction : ", disc(gen_image).item())
                
            gen_image_num = gen_image.reshape(3,128, 128).cpu()
            gen_image_num = (gen_image_num + 1) / 2  # Rescale from [-1, 1] to [0, 1]
            gen_image_num = gen_image_num.permute(1,2,0)
            # real_image_num = real[0].reshape(3,128, 128).cpu()
             
            # Plot the generated imageeeeee yee
            plt.subplot(1,2,1)
            plt.imshow(gen_image_num)
            plt.title("Generated Image")
            
            # Ploting Trained imageeee
            # plt.subplot(1,2,2)
            # plt.imshow(real_image_num, cmap="gray")
            # plt.title("Trained Image")
            
            # plt.axis("off")
            plt.show()
# %%

