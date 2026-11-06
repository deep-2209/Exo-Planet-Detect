import argparse
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

p = argparse.ArgumentParser()
p.add_argument("--bucket", required=True, help="terraform output bucket_name")
p.add_argument("--role_arn", required=True, help="terraform output sagemaker_role_arn")
p.add_argument("--region", default="us-east-1")
args = p.parse_args()

boto_sess = boto3.Session(region_name=args.region)
sm_sess = sagemaker.Session(boto_session=boto_sess, default_bucket=args.bucket)

# 1. Upload the dataset to S3 as the "training" channel
train_s3 = sm_sess.upload_data(path="dataset.npz", bucket=args.bucket,
                               key_prefix="exo-detect/input")
print("Uploaded data to:", train_s3)

# 2. Configure the managed-PyTorch training job (script mode)
estimator = PyTorch(
    entry_point="train.py",          # your script
    source_dir="sagemaker",          # folder with train.py + requirements.txt
    role=args.role_arn,              # the Terraform-created execution role
    framework_version="2.2",         # managed PyTorch container version
    py_version="py310",
    instance_type="ml.m5.large",     # CPU; ~$0.12/hr, job is a few minutes
    instance_count=1,
    hyperparameters={"epochs": 150, "batch": 16, "seeds": 5},
    output_path=f"s3://{args.bucket}/exo-detect/output",
    base_job_name="exo-detect-cnn",
    sagemaker_session=sm_sess,
)

# 3. Run it — blocks and streams CloudWatch logs to your console
estimator.fit({"training": train_s3})
print("Done. Model artifact:", estimator.model_data)