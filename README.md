# DAVOS - Diffusion Auto-Vocabulary Segmentation

Creating high-definition semantic segmentation masks from BLIP Cluster Captions using Diffusion Models

## Getting Started

ToDo: Specify the exact requirements for this project.

1. **System Requirements**: Ensure you have access to a machine with a GPU with CUDA support. This code is tested on NVIDIA A100 GPUs.
2. **Python Version**: The code is tested with Python 3.7

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ryan-Ott/DAVOS.git
```

### Step 2: Set Up the Environment

To create the conda environment, navigate to the project directory and run the following command:

```bash
conda env create -f davos_env.yml
```

**Note**: For on snellius, you can run the job script at `jobs/scripts/davos_env_setup.job` to handle the environment creation

### Step 1: Download the Data

Download the dataset from [Dataset Source](#).

Place the downloaded data in a directory named `dataset` as follows:

```text
DAVOS/
├── modules/
│   ├── PIDM/  <-- ToDo: rename this
│   │   ├── dataset/
│   │   │   ├── PASCAL_VOC/
│   │   │   │   ├── img/
│   │   │   │   ├── pose/  <-- ToDo: rename this
│   │   │   │   ├── train_pairs.txt
│   │   │   │   ├── test_pairs.txt
```

## Step 2: Prepare the Data

Run the `prepare_data.py` script to preprocess the data.

```bash
python modules/PIDM/data/prepare_data.py \
--input_dir ./dataset/PASCAL_VOC \
--output_dir ./prepared_data \
--sizes ((256,256),)
```

This will create an lmdb dataset `modules/PIDM/dataset/PASCAL_VOC/256-256/`.
This will preprocess the raw data and save it in the `prepared_data` directory.

## Step 3: Set Up the Environment

You have two options for setting up the environment: using a provided environment file or creating your own.

### Option 1: Using Provided Environment File

1. Create a new conda environment using the provided `environment.yml` file.

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment.

   ```bash
   conda activate myenv
   ```

### Option 2: Creating Your Own Environment

1. Create a new conda environment.

   ```bash
   conda create --name myenv python=3.8
   ```

2. Activate the environment.

   ```bash
   conda activate myenv
   ```

3. Install the necessary packages. You can use `pip` or `conda` to install these packages. Here is a sample list of packages. Adjust as necessary based on your requirements.

   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should list all necessary packages. You can generate it from your current environment using:

   ```bash
   pip freeze > requirements.txt
   ```

## Step 4: Set Up Weights and Biases (wandb)

1. Create a Weights and Biases account at [wandb](https://wandb.ai/).
2. Initialize wandb in the terminal.

   ```bash
   wandb login
   ```

3. Ensure your script is set up to use wandb. If not, add the following snippet to initialize wandb in your `train.py` script.

   ```python
   import wandb
   wandb.init(project="YourProjectName", entity="YourWandbEntity")
   ```

## Step 5: Run the Model

1. Ensure you are in the correct environment.

   ```bash
   conda activate myenv
   ```

2. Run the training script on the dataset.

   ```bash
   python train.py --exp_name your_experiment_name --DiffConfigPath ./config/diffusion.conf --DataConfigPath ./config/data.yaml --dataset_path ./prepared_data --save_path ./checkpoints --batch_size 8 --n_gpu 1 --epochs 300
   ```

3. For multi-GPU setup, you need to use `torch.distributed.launch`. Modify the script execution command as needed.

   ```bash
   python -m torch.distributed.launch --nproc_per_node=8 train.py --exp_name your_experiment_name --DiffConfigPath ./config/diffusion.conf --DataConfigPath ./config/data.yaml --dataset_path ./prepared_data --save_path ./checkpoints --batch_size 64 --n_gpu 8 --epochs 300
   ```

## Additional Notes

- **Checkpoint Saving**: Model checkpoints will be saved in the `checkpoints` directory.
- **Logging**: Logs and sample images will be logged to wandb.

## Troubleshooting

If you encounter any issues, ensure:

1. All paths are correctly set.
2. The environment is properly activated.
3. All necessary packages are installed.

For further assistance, refer to the documentation or contact the project maintainer.

---

This README template should cover the essential steps required to run your project. Adjust the instructions based on your specific project requirements and environment setup.
