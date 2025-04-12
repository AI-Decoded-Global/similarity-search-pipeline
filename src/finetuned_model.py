"""
Functions for generating embeddings with fine-tuned model from volunteer descriptions.
"""
import os
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import Dataset
from torch.utils.data import DataLoader

def finetune_model(finetuned_model_name):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the training examples based on volunteer profiles
    training_examples = [
        # Similar pairs - High similarity labels
        InputExample(texts=[
            "software developer (Python, JS, some C++) and looking to help with open source projects", 
            "Web developer - HTML, CSS, WordPress, PHP. Will help small community orgs with websites"
        ], label=0.9),
        
        InputExample(texts=[
            "retired nurse here. 30+ years experience. want to help in underserved areas with basic healthcare", 
            "physical therapist here. can offer services 1 day/week to seniors or others who need it"
        ], label=0.85),
        
        InputExample(texts=[
            "passionate about saving the environment!!! organized 3 beach clean-ups last year", 
            "environmental scientist (phd). can help with monitoring programs, water testing, etc"
        ], label=0.85),
        
        InputExample(texts=[
            "Former teacher, taught middle school for 15 years. Want to tutor kids who need help", 
            "early childhood educator here! love working with little ones on reading & writing"
        ], label=0.8),
        
        InputExample(texts=[
            "hablo español e inglés!!! Me encantaría trabajar como traductor para comunidades de inmigrantes", 
            "Vécu en Afrique pendant 10 ans, je parle couramment français, anglais et un peu d'arabe. Peut aider avec la traduction"
        ], label=0.8)]

    augmented_examples = [
        # Dissimilar pairs - Low similarity labels
        InputExample(texts=[
            "database admin & excel wizard. can help orgs manage their data, create reports etc", 
            "carpenter 15+ yrs, have my own tools. can help build stuff for housing projects etc"
        ], label=0.2),
        
        InputExample(texts=[
            "artist and art teacher! love doing workshops with kids!", 
            "Accountant with 10+ yrs exp. Can help small orgs with their books"
        ], label=0.2),
        
        InputExample(texts=[
            "CHEF with lots of experience! would love to teach cooking basics to people", 
            "i'm a lawyer (corporate law background) & can give some free legal advice to nonprofits"
        ], label=0.1),
        
        InputExample(texts=[
            "wilderness guide during summers. Can lead outdoor education hikes, teach survival skills", 
            "digital marketing is my thing! SEO, PPC, email campaigns etc"
        ], label=0.1),
        
        InputExample(texts=[
            "First aid instructor (certified). Looking to volunteer with disaster relief orgs when needed", 
            "MUSICIAN!! play guitar, drums & keyboard. happy to perform at fundraisers or teach basic music"
        ], label=0.1)
    ]

    all_examples = training_examples + augmented_examples
    print(f"Total number of training examples: {len(all_examples)}")


    # Define batch size and epochs
    batch_size = 8
    num_epochs = 15

    # Create DataLoader directly from examples list
    train_dataloader = DataLoader(all_examples, shuffle=True, batch_size=batch_size)

    # Load the model
    model_name = 'all-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    model.to(device)

    # Define the loss function
    train_loss = losses.CosineSimilarityLoss(model)

    # Set up warmup steps
    warmup_steps = int(len(train_dataloader) * 0.1)  # 10% of training data

    # Define output directory
    output_dir = finetuned_model_name  # "volunteer-embeddings-model"
    os.makedirs(output_dir, exist_ok=True)

    # Train the model
    print(f"Starting fine-tuning of {model_name} model...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True
    )

    print(f"Model fine-tuning complete. Saved to {output_dir}")
