CURR_DIR=$(pwd)

# Location of data
TRAIN_IMG_DIR="${CURR_DIR}/data/train"
TRAIN_CAPTIONS="${CURR_DIR}/data/train/train_captions"
TEST_IMG_DIR="${CURR_DIR}/data/test"
TEST_CAPTIONS="${CURR_DIR}/data/test/test_captions"

# CNN params
CNN_MODEL_FILE="${CURR_DIR}/pretrained/vgg16_weights.npz"

# Training parameters
NUM_EPOCHS=1000
BATCH_SIZE=20
SAVE_FREQ=200

python main.py --train_img_dir "${TRAIN_IMG_DIR}"	\
		--train_captions "${TRAIN_CAPTIONS}"	\
		--test_img_dir "${TEST_IMG_DIR}"	\
		--test_captions "${TEST_CAPTIONS}"	\
       	--cnn_model_file "${CNN_MODEL_FILE}" \
              --num_epochs ${NUM_EPOCHS}  \
              --batch_size ${BATCH_SIZE}  \
              --ckpt_freq ${SAVE_FREQ}    \
              --train
       	# --train_cnn	\
       # [--ckpt_dir CKPT_DIR] 
       # [--ckpt_freq CKPT_FREQ]
       # [--solver SOLVER] 
       # [--learning_rate LEARNING_RATE]
       # [--batch_size BATCH_SIZE] 
       # [--num_epochs NUM_EPOCHS]
       # [--cnn_model CNN_MODEL]
       # [--hidden_size HIDDEN_SIZE]
       # [--dim_embed DIM_EMBED] 
       # [--dim_decoder DIM_DECODER]