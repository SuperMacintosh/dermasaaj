import os

##################  VARIABLES  ##################

IMAGE_HEIGHT=int(os.environ.get('IMAGE_HEIGHT'))
IMAGE_WIDTH=int(os.environ.get('IMAGE_WIDTH'))
BATCH_SIZE=int(os.environ.get('BATCH_SIZE'))
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_TARGET = os.environ.get("MODEL_TARGET")

##################  CONSTANTS  #####################
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "project_outputs")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "project_outputs")
