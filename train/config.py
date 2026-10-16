import sys
from train.utils import get_default_device

from dataclasses import dataclass,asdict
import os
import pprint
import pickle


@dataclass
class BaseConfig:
    repo_home_dir: str = os.environ.get("CLEAR_REPO_DIR", os.getcwd())
    DEVICE : str = get_default_device()
    ds_name : str = 'ISIC_2018'

    split: str ='train' # 'train', 'test'
    record_dir: str = os.environ.get("CLEAR_RECORDS_ROOT", "./records")
    DATASET: str
    if (ds_name.startswith('OUT')):
        DATASET: str = ds_name
    else: DATASET: str = ds_name+'-'+split  #  "MNIST"
    

    # For logging inferece images and saving checkpoints.
    root_log_dir: str = os.path.join(record_dir,DATASET,"Logs_Checkpoints", "Inference")
    root_checkpoint_dir:str = os.path.join(record_dir,DATASET,"Logs_Checkpoints", "checkpoints")
    root_prevs_dir:str = os.path.join(record_dir,DATASET,'PreVs')


    # Current log and checkpoint directory.
    log_dir:str = "version_0"
    checkpoint_dir:str = "version_0"



@dataclass
class TrainingConfig:
    IMG_SHAPE: tuple[int]
    BATCH_SIZE: int
    SAMPLE_NUM: int
    NUM_EPOCHS: int

    if ('CHEXPERT' in BaseConfig.DATASET) or ('NIH' in BaseConfig.DATASET) :
        IMG_SHAPE = (1,224,224)
        BATCH_SIZE = 16
        SAMPLE_NUM = min(BATCH_SIZE*4,32)
        NUM_EPOCHS = 500

    elif 'Che_PM' in BaseConfig.DATASET:
        IMG_SHAPE = (1, 512, 512)
        BATCH_SIZE = 1
        SAMPLE_NUM = min(BATCH_SIZE * 4, 32)
        NUM_EPOCHS = 500
    elif 'ISIC' in BaseConfig.DATASET:
        # Select the image size based on the dataset name.
        if '512' in BaseConfig.DATASET or 'ISIC_2018' in BaseConfig.DATASET or 'ISIC_Ruler' in BaseConfig.DATASET:
            IMG_SHAPE = (3, 512, 512)
            BATCH_SIZE = 1  # Use a smaller batch size to limit GPU memory usage for 512x512 images.
        elif '1024' in BaseConfig.DATASET:
            IMG_SHAPE = (3, 1024, 1024)
            BATCH_SIZE = 1  # Local tests with 1024x1024 images require a batch size of 1.
        else:
            IMG_SHAPE = (3, 224, 224)
            BATCH_SIZE = 16
        SAMPLE_NUM = min(BATCH_SIZE*4, 32)
        
        if IMG_SHAPE[2] == 224:
            NUM_EPOCHS = 500
        elif IMG_SHAPE[2] == 512 or IMG_SHAPE[2] == 1024:
            NUM_EPOCHS = 500
        elif IMG_SHAPE[2] == 64:
            NUM_EPOCHS = 200
        else:
            print(f'{NUM_EPOCHS=}')
    else:
        raise Exception(f'Not implemented {BaseConfig.DATASET=}')
    LR : float = 2e-4
    NUM_WORKERS : int = 2



def save_config_train_diffusion(tc: TrainingConfig,
                                bc: BaseConfig,
                                save_dir: str,
                                model_info: dict = None):
    '''
    save the config of the velocity-model training (model_info = DiT arch dict)
    '''
    # save as dict
    config_dict = {}
    config_dict['BaseConfig'] = asdict(bc)
    config_dict['ModelConfig'] = dict(model_info or {})
    config_dict['TrainingConfig'] = asdict(tc)

    with open(save_dir+'/config.pickle', 'wb') as handle:
        pickle.dump(config_dict, handle)

    # save to txt
    config_str = pprint.pformat(config_dict,sort_dicts=False)
    with open(save_dir+'/config.txt', "w") as f:
        f.write(config_str)

    return

