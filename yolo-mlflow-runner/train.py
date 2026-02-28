import os
import json
import shutil
from pathlib import Path

import mlflow
from ultralytics import YOLO
import albumentations as A
import cv2


def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v if v is not None else default


def to_int(name: str, default: int) -> int:
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def to_float(name: str, default: float) -> float:
    try:
        return float(env(name, str(default)))
    except ValueError:
        return default


def to_bool_str(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def copytree_safe(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    # Python <3.8 compat not needed; but keep safe overwrite behavior
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        out = dst / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def main():
    # ---- MLflow config ----
    tracking_uri = env("MLFLOW_TRACKING_URI")
    exp_name = env("MLFLOW_EXPERIMENT_NAME", "yolo_runs")
    run_name = env("MLFLOW_RUN_NAME", "")

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required (point to Machine 1).")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    # ---- Training config ----
    data_yaml = env("DATA_YAML", "/data/dataset.yaml")
    model_name = env("MODEL", "yolov8n.pt")
    task = env("TASK", "detect")
    epochs = to_int("EPOCHS", 10)
    imgsz = to_int("IMGSZ", 640)
    batch = to_int("BATCH", "auto")
    device = env("DEVICE", "0")
    workers = to_int("WORKERS", 8)
    seed = to_int("SEED", 0)

    project_dir = Path(env("PROJECT_DIR", "/outputs/ultralytics"))
    export_dir = Path(env("EXPORT_DIR", "/outputs/exported"))
    project_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Optional extension point
    extra_args = {}
    try:
        extra_args = json.loads(env("EXTRA_ARGS_JSON", "{}"))
        if not isinstance(extra_args, dict):
            extra_args = {}
    except json.JSONDecodeError:
        extra_args = {}

    # Log params consistently
    params = {
        "data_yaml": data_yaml,
        "model": model_name,
        "task": task,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "seed": seed,
        "extra_args_json": json.dumps(extra_args, sort_keys=True),
    }

    # ---- Run ----
    with mlflow.start_run(run_name=run_name or None) as run:
        mlflow.log_params(params)

        # Ensure deterministic-ish where possible
        # Ultralytics accepts seed in train()
        yolo = YOLO(model_name)

        train_kwargs = dict(
            data=data_yaml,
            task=task,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            seed=seed,
            project=str(project_dir),
            name=f"run_{run.info.run_id[:8]}",
            exist_ok=True,
            custom_transforms=[
                # -------------------------
                # Flips (both directions)
                # -------------------------
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # -------------------------
                # Perspective / drone tilt
                # -------------------------
                A.Perspective(scale=(0.03, 0.12), keep_size=True, p=0.5),
                A.Affine(
                    scale=(0.75, 1.25),
                    translate_percent=(0.0, 0.12),
                    rotate=(-20, 20),
                    shear=(-10, 10),
                    mode=cv2.BORDER_CONSTANT,
                    p=0.7,
                ),
                # Make objects small (altitude simulation)
                A.RandomResizedCrop(
                    size=(imgsz, imgsz),
                    scale=(0.25, 1.0),
                    ratio=(0.75, 1.33),
                    p=0.6,
                ),
                # -------------------------
                # Motion (object + drone)
                # -------------------------
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=(9, 35)),  # strong linear motion
                        A.GaussianBlur(blur_limit=(3, 9)),
                    ],
                    p=0.6,
                ),
                # Add directional smear feel
                A.Downscale(scale_min=0.5, scale_max=0.85, p=0.35),
                # -------------------------
                # Dust / haze simulation
                # -------------------------
                A.OneOf(
                    [
                        A.RandomFog(
                            fog_coef_lower=0.08,
                            fog_coef_upper=0.35,
                            alpha_coef=0.1,
                        ),
                        A.RandomShadow(
                            num_shadows_lower=1,
                            num_shadows_upper=3,
                            shadow_dimension=5,
                        ),
                    ],
                    p=0.35,
                ),
                # Noise (sensor + environment)
                A.OneOf(
                    [
                        A.ISONoise(color_shift=(0.02, 0.1), intensity=(0.2, 0.8)),
                        A.GaussNoise(var_limit=(20.0, 120.0)),
                    ],
                    p=0.4,
                ),
                # Lighting shifts
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.5,
                ),
                A.RandomGamma(gamma_limit=(70, 150), p=0.3),
                # Compression artifacts (radio transmission feel)
                A.ImageCompression(
                    quality_lower=20,
                    quality_upper=65,
                    p=0.5,
                ),
            ],
        )
        train_kwargs.update(extra_args)

        results = yolo.train(**train_kwargs)

        # Ultralytics returns a Results object; metrics are usually in results.results_dict
        results_dict = getattr(results, "results_dict", None) or {}
        # Log metrics (MLflow expects float-ish values)
        for k, v in results_dict.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass

        # Locate Ultralytics run directory
        # Commonly: {project}/{name}
        run_dir = project_dir / train_kwargs["name"]
        if not run_dir.exists():
            # Fallback: Ultralytics sometimes writes to runs/{task}/{name}
            alt = Path("runs") / task / train_kwargs["name"]
            if alt.exists():
                run_dir = alt

        # Log key artifacts to MLflow
        if run_dir.exists():
            mlflow.log_artifacts(str(run_dir), artifact_path="ultralytics_run")

        # Export/copy to local volume for retention
        local_export = export_dir / f"{run.info.run_id}"
        copytree_safe(run_dir, local_export)

        # Also store a minimal manifest
        manifest = local_export / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "mlflow_tracking_uri": tracking_uri,
                    "experiment": exp_name,
                    "run_id": run.info.run_id,
                    "params": params,
                    "run_dir": str(run_dir),
                    "export_dir": str(local_export),
                },
                indent=2,
            )
        )

        print(f"MLflow run_id: {run.info.run_id}")
        print(f"Local export: {local_export}")


if __name__ == "__main__":
    main()
