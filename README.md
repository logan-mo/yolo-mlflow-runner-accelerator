docker run --rm --gpus all --shm-size=16g \
    -e MLFLOW_EXPERIMENT_NAME="ultralytics_yolo" \
    -e DATA_YAML="/data/data.yaml" \
    -e MLFLOW_RUN_NAME="yolo1024speedtest" \
    -e MODEL="yolo26n.pt" \
    -e TASK="detect" \
    -e EPOCHS="300" \
    -e BATCH="64" \
    -e WORKERS="128" \
    -v /home/ubuntu/Desktop/Army_Project_Trainings/data/tank-component-26th_feb_out_5be890227660:/data:ro \
    -v /home/ubuntu/Desktop/Army_Project_Trainings/runs:/outputs \
    yolo-mlflow-runner:latest