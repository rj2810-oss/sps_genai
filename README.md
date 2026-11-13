# sps_genai: Applied Generative AI, Assignment 1

FastAPI service with two models:

- Bigram text generation (Module 3 class activity): `POST /generate`
- spaCy word embeddings (Assignment 1): `POST /embedding` and `POST /similarity`, using the 300-dimensional `en_core_web_lg` vectors from the Module 2 demo

## Run with Docker
```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```
Interactive docs: http://127.0.0.1:8000/docs

## Run locally without Docker
```bash
uv sync
uv run fastapi dev main.py
```

## Example requests
```bash
# Word embedding
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" -d '{"word": "king"}'

# Similarity between two words
curl -X POST http://127.0.0.1:8000/similarity \
  -H "Content-Type: application/json" -d '{"word1": "king", "word2": "queen"}'

# Bigram text generation
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" -d '{"start_word": "the", "length": 10}'
```

Words without a vector return 404. Multi-word input returns the average of the word vectors.

## Assignment 2: CNN Image Classifier

Added a CNN that classifies images into the 10 CIFAR-10 classes, plus a new `/classify` endpoint in the API.

### What's new

- `app/cnn_model.py` - the CNN (2 conv + pooling layers, then 2 fully connected layers, 64x64x3 input, 10 outputs)
- `app/image_classifier.py` - loads the trained weights and runs predictions
- `app/models/cnn_cifar10.pt` - trained weights (best test accuracy 70.42%)
- `train_cnn.py` - script used to train the model on CIFAR-10

### Running it

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs and try `POST /classify`.

### Classifying an image

```bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@your_image.jpg;type=image/jpeg"
```

Accepts png, jpg and webp. Returns the predicted class, the confidence, and the top 3 classes.

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Retraining

```bash
uv run python train_cnn.py --epochs 10
```

This downloads CIFAR-10 (about 170 MB) into `data/`, which is gitignored.

## Assignment 3: GAN for MNIST digit generation

A GAN trained on MNIST that generates hand-written digits, served through the `/generate_digits` endpoint.

**Files**
- `app/gan_model.py`: Generator and Discriminator architecture
- `app/gan_generator.py`: loads trained weights and generates images
- `app/generator.pth`: trained generator weights
- `train_gan.py`: training script (20 epochs, Adam, BCE loss)

**Run with Docker**
```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

**Generate digits**
Open http://127.0.0.1:8000/docs and try GET /generate_digits, or:
```bash
curl "http://127.0.0.1:8000/generate_digits?num_images=16" --output digits.png
```
Parameters: `num_images` (1 to 64, default 16), `seed` (optional, for reproducible output).

**Retrain (optional)**
```bash
uv run python train_gan.py
```

## Assignment 4: Diffusion and Energy-Based Models for CIFAR-10

Two image generators trained on CIFAR-10, served through `/generate_diffusion` and `/generate_ebm`.

**Files**
- `app/diffusion_model.py`: sinusoidal time embedding, UNet noise predictor, DDPM forward/reverse process (T = 500)
- `app/ebm_model.py`: CNN energy function, Langevin sampler (gradient descent on the input image with model weights frozen), replay buffer
- `app/cifar_generators.py`: loads trained weights and generates PNG grids for the API
- `app/diffusion.pth`, `app/ebm.pth`: trained weights
- `train_diffusion.py`, `train_ebm.py`: training scripts

**Training**
- Diffusion: 15 epochs, batch size 128, Adam (lr 2e-4), MSE loss on predicted noise. Final average loss about 0.048.
- EBM: 3 epochs, batch size 128, Adam (lr 1e-4), contrastive divergence loss with energy regularization, 60 Langevin steps per batch, replay buffer of 8,192 images.

**Run with Docker**
```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

**Generate images**
Open http://127.0.0.1:8000/docs and try GET /generate_diffusion or GET /generate_ebm, or:
```bash
curl "http://127.0.0.1:8000/generate_diffusion?num_images=8" --output diffusion.png
curl "http://127.0.0.1:8000/generate_ebm?num_images=8&steps=256" --output ebm.png
```

**Retrain (optional)**
```bash
uv run python train_diffusion.py --epochs 15
uv run python train_ebm.py --epochs 3
```