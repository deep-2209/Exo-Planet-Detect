terraform {
  required_version = ">= 1.5"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.0" }
    random = { source = "hashicorp/random", version = "~> 3.0" }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

# Globally-unique bucket name needs a suffix so it won't collide
resource "random_id" "suffix" {
  byte_length = 4
}

# ---------------- S3: dataset input + model/metrics artifacts ----------------
resource "aws_s3_bucket" "artifacts" {
  bucket        = "${var.project}-${data.aws_caller_identity.current.account_id}-${random_id.suffix.hex}"
  force_destroy = true   # so `terraform destroy` empties + removes the bucket in one command

  tags = {
    Project = var.project
    Purpose = "SageMaker training data and artifacts"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------- IAM: role SageMaker assumes to run the training job --------
data "aws_iam_policy_document" "sm_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_exec" {
  name               = "${var.project}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sm_assume.json
  tags               = { Project = var.project }
}

# Least-privilege: this bucket, CloudWatch Logs, and ECR read (to pull the
# managed PyTorch training image). Fallback note below if a job hits AccessDenied.
data "aws_iam_policy_document" "sm_perms" {
  statement {
    sid       = "S3Bucket"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
    resources = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream",
                 "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["arn:aws:logs:*:*:log-group:/aws/sagemaker/*"]
  }
  statement {
    sid       = "ECRPull"
    actions   = ["ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability",
                 "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"]
    resources = ["*"]   # GetAuthorizationToken requires *; the rest is read-only image pull
  }
}

resource "aws_iam_role_policy" "sagemaker_exec" {
  name   = "${var.project}-sagemaker-exec-policy"
  role   = aws_iam_role.sagemaker_exec.id
  policy = data.aws_iam_policy_document.sm_perms.json
}