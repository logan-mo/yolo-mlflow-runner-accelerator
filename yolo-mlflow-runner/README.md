MLFLOW_EXPERIMENT_NAME="" MLFLOW_RUN_NAME="" MODEL="" TASK="" EPOCHS=""

docker build -t yolo-mlflow-runner:latest .

docker run --rm --gpus all \
  -e MLFLOW_EXPERIMENT_NAME="ultralytics_yolo" \
  -e MLFLOW_RUN_NAME="tank_dataset" \
  -e DATA_YAML="/data/data.yaml" \
  -e MODEL="yolo26n-seg.pt" \
  -e TASK="segment" \
  -e EPOCHS="1" \
  -e BATCH="16" \
  -e WORKERS="1" \
  -v /home/ubuntu/Desktop/Army_Project_Trainings/data/bunker_openings_dataset_out_a1963bcaf595:/data:ro \
  -v /home/ubuntu/Desktop/Army_Project_Trainings/runs:/outputs \
  yolo-mlflow-runner:latest


docker run --rm --gpus all --shm-size=16g\
  -e MLFLOW_EXPERIMENT_NAME="ultralytics_yolo" \
  -e MLFLOW_RUN_NAME="tank_dataset" \
  -e DATA_YAML="/data/data.yaml" \
  -e MODEL="yolo26n.pt" \
  -e TASK="detect" \
  -e EPOCHS="300" \
  -e BATCH="128" \
  -e WORKERS="8" \
  -v /home/ubuntu/Desktop/Army_Project_Trainings/data/fixed_dataset_tanks_only:/data:ro \
  -v /home/ubuntu/Desktop/Army_Project_Trainings/runs:/outputs \
  yolo-mlflow-runner:latest

