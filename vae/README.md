# Selective Amnesia (SA) for VAEs

# Requirements

Install the requirements using a `conda` environment:
```
conda create --name sa-vae python=3.8
conda activate sa-vae
pip install -r requirements.txt
```

# Forgetting Training with SA

1. First train a conditional VAE on all 10 MNIST classes.

    ```
    CUDA_VISIBLE_DEVICES="0" python train_cvae.py --config mnist.yaml --data_path ./dataset
    ```

    where you should pass the GPU ID into `CUDA_VISIBLE_DEVICES` for GPU training, or pass an empty string for CPU training.
    The checkpoint should be saved in a folder that looks like `results/yyyy_mm_dd_hhmmss`, where `yyyy_mm_dd_hhmmss` corresponds
    to the date and time that your training run was started.

2. Calculate the FIM.

    ```
    CUDA_VISIBLE_DEVICES="0" python calculate_fim.py --ckpt_folder results/yyyy_mm_dd_hhmmss
    ```
    
    The FIM should be saved as `fisher_dict.pkl` in the same folder.

3. Forgetting training with SA

    ```
    CUDA_VISIBLE_DEVICES="0" python train_forget.py --ckpt_folder results/yyyy_mm_dd_hhmmss --label_to_drop 0 --lmbda 100
    ```

    Another folder `results/yyyy_mm_dd_hhmmss` will be created with the forgetting VAE checkpoint and logging samples stored.

# Classifier Evaluation

1. First train a classifier on the MNIST dataset

    ```
    CUDA_VISIBLE_DEVICES="0" python train_classifier.py --data_path ./dataset
    ```

    The classifier checkpoint should be saved in the folder `classifier_ckpts`.

2. Generate samples of the forgetting class on the VAE checkpoint trained with SA.

    ```
    CUDA_VISIBLE_DEVICES="0" python generate_samples.py --ckpt_folder results/yyyy_mm_dd_hhmmss --label_to_generate 0 --n_samples 1000
    ```
    
    The samples should be stored in `results/yyyy_mm_dd_hhmmss/0_samples`.

3. Evaluate with the classifier

    ```
    CUDA_VISIBLE_DEVICES="0" python evaluate_with_classifier.py --sample_path results/yyyy_mm_dd_hhmmss --label_of_dropped_class 0
    ```

    The average probability and average entropy should be printed to your screen as follows

    ```
    Average entropy: 2.194569170475006  
    Average prob of forgotten class: 0.058080864138901234
    ```