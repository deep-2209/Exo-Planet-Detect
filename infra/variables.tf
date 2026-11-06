variable "region" {
  description = "AWS region for the Exo-Detect training pipeline"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name, used to name and tag resources"
  type        = string
  default     = "exo-detect"
}