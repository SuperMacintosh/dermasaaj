from preprocessing import initialize_dataset_from_file, get_split_image_data
from model import initialize_model, compile_model, train_model, evaluate_model
import os
#initialize tensorflow dataset
url_file_name="https://storage.googleapis.com/derma-data/raw_data/archive.zip"
parent_path=initialize_dataset_from_file(url_file_name,extract=True,archive_format='zip')


# calibrate data structure
img_height=int(os.getenv('IMAGE_HEIGHT'))
img_width=int(os.getenv('IMAGE_WIDTH'))
batch_size=int(os.getenv('BATCH_SIZE'))

# Split and prepare inputs for the model
child_path='data/train'
train_ds=get_split_image_data(parent_path, child_path, img_height, img_width,batch_size )
child_path='data/test'
test_ds=get_split_image_data(parent_path, child_path, img_height, img_width,batch_size )
child_path='data/valid'
val_ds=get_split_image_data(parent_path, child_path, img_height, img_width,batch_size )

# initialize and finetune the CNN model
num_classes=len(train_ds.class_names)
kernel_size=3
val_dropout=0.2
model=initialize_model(num_classes, kernel_size, val_dropout,img_height, img_width)
# compile the model
model=compile_model(model)
#fit the model
patience=2
verbose=1
model, history=train_model(
        model,
        train_ds,
        val_ds,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose
        )
