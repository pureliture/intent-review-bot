terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  skip_region_validation      = true
  s3_use_path_style           = true

  endpoints {
    ec2            = "http://127.0.0.1:4566"
    dynamodb       = "http://127.0.0.1:4566"
    events         = "http://127.0.0.1:4566"
    iam            = "http://127.0.0.1:4566"
    lambda         = "http://127.0.0.1:4566"
    logs           = "http://127.0.0.1:4566"
    s3             = "http://127.0.0.1:4566"
    secretsmanager = "http://127.0.0.1:4566"
    sts            = "http://127.0.0.1:4566"
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

locals {
  lambda_zip_path = abspath("${path.module}/../../dist/lambda.zip")
}

module "on_demand" {
  source = "../.."

  name_prefix             = "tf-ondemand-eval"
  lambda_zip_path         = local.lambda_zip_path
  worker_ami_id           = var.worker_ami_id
  worker_instance_type    = "t3.small"
  vpc_id                  = data.aws_vpc.default.id
  subnet_id               = sort(data.aws_subnets.default.ids)[0]
  allowed_installations   = []
  allowed_repositories    = []
  max_vm_seconds          = 1800
  session_expiration_days = 30
  create_log_groups       = false
}

variable "worker_ami_id" {
  description = "A LocalStack-only seeded image ID; no image is built or booted by this example."
  type        = string
  default     = "ami-1e749f67"
}
