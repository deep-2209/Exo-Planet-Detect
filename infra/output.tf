output "bucket_name" {
  description = "S3 bucket for training data and artifacts"
  value       = aws_s3_bucket.artifacts.bucket
}

output "sagemaker_role_arn" {
  description = "IAM role ARN the SageMaker training job assumes"
  value       = aws_iam_role.sagemaker_exec.arn
}