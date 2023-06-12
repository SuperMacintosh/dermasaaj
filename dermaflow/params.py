import os

##################  VARIABLES  ##################

IMAGE_HEIGHT=int(os.environ.get('IMAGE_HEIGHT'))
IMAGE_WIDTH=int(os.environ.get('IMAGE_WIDTH'))
BATCH_SIZE=int(os.environ.get('BATCH_SIZE'))
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
PATH_URL_ARCHIVE_FILE=os.environ.get("PATH_URL_ARCHIVE_FILE")
ARCHIVE_EXTRACT=os.environ.get("ARCHIVE_EXTRACT")
ARCHIVE_FILE=os.environ.get("ARCHIVE_FILE")
ARCHIVE_PARENT_FOLDER=os.environ.get("ARCHIVE_PARENT_FOLDER")

IMAGES_CROP=os.environ.get("IMAGES_CROP")
IMAGES_AUGMENT=os.environ.get("IMAGES_AUGMENT")
OUTPUT_PARENT_FOLDER=os.environ.get("OUTPUT_PARENT_FOLDER")
MODEL_TYPE=os.environ.get("MODEL_TYPE").strip(' ')

##################  CONSTANTS  #####################
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "project_outputs")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "project_outputs")
LOCAL_CHECKPOINT_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "project_outputs","checkpoint")
