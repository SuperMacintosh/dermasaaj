from dermaflow.logic.preprocessing import initialize_dataset_from_file, get_split_image_data,densenet201_preprocess
from dermaflow.logic.model import initialize_model, compile_model, train_model, evaluate_model
from dermaflow.logic.registry import save_model
from pathlib import Path
from dermaflow.params import *

# Extract if need be dataset from archive
# original or cropped and augmented dataset


if str.upper(ARCHIVE_EXTRACT) == 'YES':
    parent_path = Path("../root/.keras/datasets")

    if 0 != len(ARCHIVE_PARENT_FOLDER):
        parent_path+=f'/{ARCHIVE_PARENT_FOLDER}'

    if not parent_path.is_dir():
        file_name=f'{PATH_URL_ARCHIVE_FILE}/{ARCHIVE_FILE}'
        parent_path=initialize_dataset_from_file(file_name,extract=True,archive_format='zip')
else:
    # implement here the image processing
    if str.upper(IMAGES_CROP) == 'YES':
        # report here jeremy's codes
        print('Missing cropping codes')
    if str.upper(IMAGES_AUGMENT) == 'YES':
        # report here adama's codes
        print('Missing augmentation codes')

    parent_path=OUTPUT_PARENT_FOLDER

# initialize tensorflow dataset
# & calibrate data structure

img_height=IMAGE_HEIGHT
img_width=IMAGE_WIDTH
batch_size=BATCH_SIZE

# Split and prepare inputs for the model
child_path='train'
train_ds=get_split_image_data(parent_path, child_path, img_height, img_width,batch_size )
child_path='test'
test_ds=get_split_image_data(parent_path, child_path, img_height, img_width,batch_size )
child_path='valid'
val_ds=get_split_image_data(parent_path, child_path, img_height, img_width,batch_size )

if os.upper(MODEL_TYPE) == 'DENSENET201':
    train_ds = train_ds.map(densenet201_preprocess)
    val_ds = val_ds.map(densenet201_preprocess)
    test_ds = test_ds.map(densenet201_preprocess)


# initialize and finetune the CNN model
num_classes=len(train_ds.class_names)
kernel_size=3
val_dropout=0.2
model=initialize_model(MODEL_TYPE, num_classes, kernel_size, val_dropout,img_height, img_width)
# compile the model
model=compile_model(MODEL_TYPE,model)
#train the model
patience=2
verbose=1
model, history=train_model(
        MODEL_TYPE,
        model,
        train_ds,
        val_ds,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose
        )
# save the model
save_model(MODEL_TYPE, model)  # uses LOCAL_REGISTRY_PATH

#evaluate the model with the test data set
metrics=evaluate_model(
        model,
        test_ds,
        batch_size=BATCH_SIZE
        )
