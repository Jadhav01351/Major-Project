##########################  TEXT TO FACE GENERATION USING DCGAN AND BERT MODULE  #############################


## BERT MODULE 
 - textual input in the form of facial features is sent into the BERT module, which transforms the text into embedding vectors. The encoder and decorator in the BERT module take an input, transform it into mask symbols in the encoder, and then pass it to the decoder to create an embedding that preserves the semantic information. The d-dimensional array is formed by concatenating the BERT embeddings with random noise in order to prepare the input data for the generator.

## DC GAN
 - The DCGAN consists of two main components: the generator and the discriminator. These networks are initialized using the `Generator` and `Discriminator` classes defined in the code.

The Generator begins with a linear projection of text embeddings, reducing their dimensionality. It then utilizes transposed convolutional layers for up-sampling, progressively increasing the spatial dimensions of the input noise and projected text embeddings. Batch normalization and leaky ReLU activation functions enhance training stability and feature diversity. 

The Discriminator comprises a series of convolutional layers for down-sampling real and generated images. Additionally, it linearly projects and concatenates text embeddings, leveraging both image and text information. The discriminator's final layer produces a probability map indicating the authenticity of the input. Both Generator and Discriminator employ spectral normalization for weight stability. The overall architecture emphasizes proper weight initialization and embraces Batch Normalization for normalization. Despite not incorporating certain advanced techniques like progressive growing or attention mechanisms, the DCGAN demonstrates effectiveness in generating high-quality facial images with a focus on stability and realism.


## Loss Function Used---

   *  nn.MSELoss()
   *  nn.BCELoss()
   *  nn.L1Loss()

## Data Collection
-- The CelebA dataset with over 200,000 celebrity images is the primary data source. Data collection involves unzipping files to access images and metadata. During preprocessing, images are loaded using PIL, transformed (resized, converted to tensors, and normalized), extracting true text, and randomly selecting "wrong" images for triplet creation. Divided into training and validation sets for model learning and evaluation.
