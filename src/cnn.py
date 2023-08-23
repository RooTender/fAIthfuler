"""Provides CNN utils made for training neural network."""
import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb


class FaithfulNet(nn.Module):
    """Convolutional neural network for Faithful textures generation."""

    def __init__(self):
        super(FaithfulNet, self).__init__()
        # Upscaling layers (for example, using Transposed Convolution)
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, image):
        """Forward function of the neural network."""
        return self.upscale(image)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.model(image)


class ImageLoaderDataset(Dataset):
    def __init__(self, input_root_dir, output_root_dir, transform=None):
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.transform = transform
        self.image_paths = [(os.path.join(input_root_dir, file),
                             os.path.join(output_root_dir, file))
                            for file in os.listdir(input_root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        input_image_path, output_image_path = self.image_paths[idx]
        input_image = Image.open(input_image_path)
        output_image = Image.open(output_image_path)

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image


class CNN:
    """Trains neural networks and evaluates them."""

    def __load_data(self, input_path: str, output_path: str, batch_size: int, validation_split: float = 0.2):
        print(f'Input dataset: {input_path}')
        print(f'Output dataset: {output_path}')
        print('Reading datasets...')

        transformator = transforms.Compose([
            transforms.ToTensor()  # Convert images to tensors
        ])

        input_data = ImageLoaderDataset(
            input_path, output_path, transform=transformator)

        # Splitting the dataset into training and validation sets
        num_train = len(input_data)
        indices = list(range(num_train))
        split = int(np.floor(validation_split * num_train))

        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(input_data, batch_size=batch_size,
                                  sampler=train_sampler, pin_memory=True, num_workers=4)
        valid_loader = DataLoader(input_data, batch_size=batch_size,
                                  sampler=valid_sampler, pin_memory=True, num_workers=4)

        return train_loader, valid_loader

    def __train(self, dir_for_models: str, train_set: DataLoader, valid_set: DataLoader,
                num_of_epochs: int, learning_rate: float):

        # Initialize Generator and Discriminator
        generator = FaithfulNet().cuda()
        discriminator = Discriminator().cuda()

        # Optimizers
        generator_optim = optim.Adam(generator.parameters(), lr=learning_rate)
        discriminator_optim = optim.Adam(
            discriminator.parameters(), lr=learning_rate)

        # Losses
        criterion_gan = nn.BCELoss()

        def criterion_content(prediction, target):
            return 0.8 * nn.L1Loss()(prediction, target) + 0.2 * nn.MSELoss()(prediction, target)

        best_loss = float('inf')

        for epoch in range(num_of_epochs):
            total_real_loss = 0.0
            total_fake_loss = 0.0
            total_discriminator_loss = 0.0

            total_content_loss = 0.0
            total_generator_loss = 0.0
            total_gan_loss = 0.0

            total_batches = 0

            for (input_batch, output_batch) in tqdm(
                    train_set,
                    unit="batch",
                    desc=f'epoch {epoch}'):

                input_batch = input_batch.cuda()
                output_batch = output_batch.cuda()

                # Train Discriminator
                discriminator_optim.zero_grad()

                # Real Images
                real_output = discriminator(output_batch)
                real_loss = criterion_gan(
                    real_output, torch.ones_like(real_output).cuda())
                total_real_loss += real_loss

                # Fake Images
                fake_images = generator(input_batch)
                fake_output = discriminator(fake_images.detach())
                fake_loss = criterion_gan(
                    fake_output, torch.zeros_like(fake_output).cuda())
                total_fake_loss += fake_loss

                # Combine losses and update Discriminator
                discriminator_loss = real_loss + fake_loss
                total_discriminator_loss += discriminator_loss
                discriminator_loss.backward()
                nn.utils.clip_grad.clip_grad_norm_(
                    discriminator.parameters(), max_norm=1.0)
                discriminator_optim.step()

                # Train Generator
                generator_optim.zero_grad()

                # Content loss (MSE or L1)
                content_loss = criterion_content(fake_images, output_batch)
                total_content_loss += content_loss

                # GAN loss
                generator_fake_output = discriminator(fake_images)
                gan_loss = criterion_gan(
                    generator_fake_output, torch.ones_like(generator_fake_output).cuda())
                total_gan_loss += gan_loss

                # Combine losses and update Generator
                generator_loss = content_loss + gan_loss
                total_generator_loss += generator_loss
                generator_loss.backward()
                nn.utils.clip_grad.clip_grad_norm_(
                    generator.parameters(), max_norm=1.0)
                generator_optim.step()

                total_batches += 1

            avg_real_loss = total_real_loss / total_batches
            avg_fake_loss = total_fake_loss / total_batches
            avg_discriminator_loss = total_discriminator_loss / total_batches

            avg_content_loss = total_content_loss / total_batches
            avg_gan_loss = total_gan_loss / total_batches
            avg_generator_loss = total_generator_loss / total_batches

            avg_loss = (
                (total_generator_loss + total_discriminator_loss) / 2) / total_batches

            wandb.log({"train": {"Real images loss": avg_real_loss,
                                 "Fake images loss": avg_fake_loss,
                                 "Discriminator loss": avg_discriminator_loss,
                                 "Content loss": avg_content_loss,
                                 "GAN loss": avg_gan_loss,
                                 "Generator loss": avg_generator_loss,
                                 "Loss": avg_loss}})

            with torch.no_grad():
                total_fake_loss = 0.0
                total_real_loss = 0.0
                total_validation_loss = 0.0

                total_validation_batches = 0

                for (input_batch, output_batch) in tqdm(valid_set, unit="batch", desc=f'validating epoch {epoch}'):
                    input_batch = input_batch.cuda()
                    output_batch = output_batch.cuda()

                    # Forward pass
                    fake_images = generator(input_batch)
                    real_output = discriminator(output_batch)
                    fake_output = discriminator(fake_images.detach())

                    # Compute losses (similar to your training loop but without updates)
                    real_loss = criterion_gan(
                        real_output, torch.ones_like(real_output).cuda())
                    fake_loss = criterion_gan(
                        fake_output, torch.zeros_like(fake_output).cuda())
                    validation_loss = real_loss + fake_loss

                    total_validation_loss += validation_loss
                    total_validation_batches += 1

                avg_validation_loss = total_validation_loss / total_validation_batches
                avg_real_loss = total_real_loss / total_validation_batches
                avg_fake_loss = total_fake_loss / total_validation_batches

                wandb.log({"validation": {"Loss": avg_validation_loss,
                                          "Real images loss": avg_real_loss,
                                          "Fake images loss": avg_fake_loss}})

            if avg_loss < best_loss:
                best_loss = avg_loss

                os.makedirs(os.path.join('..', 'models',
                            dir_for_models), exist_ok=True)
                path_to_save = os.path.join('..', 'models', dir_for_models)
                torch.save(generator.state_dict(), os.path.join(
                    path_to_save, f'G_e{epoch}_{avg_generator_loss:.4f}.pth'))
                torch.save(discriminator.state_dict(), os.path.join(
                    path_to_save, f'D_e{epoch}_{avg_discriminator_loss:.4f}.pth'))

    def run(self, input_path: str, output_path: str):
        """Just a test..."""

        # SGD performs the best on 0.1
        # Adam on 0.001
        learning_rate = 0.005

        # SGD on 16
        # Adam on ???
        batch_size = 64
        epochs = 100
        optimizer = 'sgd'

        dimensions = os.path.basename(os.path.normpath(input_path))

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="FAIthfuler",
            entity="rootender",
            name=f"FaithfulNet_{dimensions}",
            # track hyperparameters and run metadata
            config={
                "architecture": "CNN",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "hidden_layer_sizes": [64],
                "optimizer": optimizer
            }
        )

        if not torch.cuda.is_available():
            print("CUDA architecture is unavailable!")
            sys.exit(1)
        else:
            print("CUDA detected! Initializing...")

        train_set, validation_set = self.__load_data(
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size
        )

        # simulate training
        self.__train(f'{dimensions}_b{batch_size}_lr{learning_rate}',
                     train_set, validation_set, epochs, learning_rate)

        wandb.finish()

    def generate_image(self, input_image_path: str, model_path="faithfulnet.pth"):
        """Generates an image using a GAN generator model."""

        # Load the FaithfulNet model and its trained weights
        model = FaithfulNet()
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()

        # Load and preprocess the input image
        input_image = Image.open(input_image_path)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        input_tensor = transform(input_image).unsqueeze(  # type: ignore
            0).cuda()

        # Generate the image
        with torch.no_grad():
            generated_image = model(input_tensor)

        # Post-process the generated image
        # Rescale from [-1, 1] to [0, 1]
        generated_image = generated_image.squeeze().cpu().detach()
        generated_image = (generated_image + 1) / 2.0
        generated_image = transforms.ToPILImage()(generated_image)
        # Convert the tensor to a PIL image (assuming it's an RGB image)

        os.makedirs(os.path.join('..', 'out'), exist_ok=True)
        generated_image.save(os.path.join(
            '..', 'out', os.path.basename(input_image_path)))
        generated_image.close()
