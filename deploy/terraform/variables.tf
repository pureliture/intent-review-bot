variable "name_prefix" {
  description = "Short stable prefix for resources."
  type        = string
  default     = "review-bot"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.name_prefix))
    error_message = "name_prefix must be 2-21 lowercase alphanumeric or hyphen characters."
  }
}

variable "lambda_zip_path" {
  description = "Path to the prebuilt Python 3.11 Lambda ZIP; Terraform does not build it."
  type        = string
  validation {
    condition     = fileexists(var.lambda_zip_path)
    error_message = "lambda_zip_path must name an existing ZIP file."
  }
}

variable "worker_ami_id" {
  description = "Operator-built Ubuntu 24.04 worker AMI."
  type        = string
}

variable "worker_instance_type" {
  description = "EC2 type used for one-shot workers."
  type        = string
  default     = "t3.large"
}

variable "vpc_id" {
  description = "Existing VPC for workers; this module does not create a VPC."
  type        = string
}

variable "subnet_id" {
  description = "Existing subnet for workers; this module does not create a subnet."
  type        = string
}

variable "allowed_installations" {
  description = "GitHub App installation IDs accepted by intake."
  type        = list(number)
  default     = []
}

variable "allowed_repositories" {
  description = "GitHub full_name values accepted by intake."
  type        = list(string)
  default     = []
}

variable "max_vm_seconds" {
  type    = number
  default = 3600
  validation {
    condition     = var.max_vm_seconds >= 60 && var.max_vm_seconds <= 21600
    error_message = "max_vm_seconds must be between 60 and 21600."
  }
}

variable "session_expiration_days" {
  description = "Lifecycle expiry for session/artifact bucket objects."
  type        = number
  default     = 30
  validation {
    condition     = var.session_expiration_days >= 1
    error_message = "session_expiration_days must be at least 1."
  }
}

variable "log_retention_days" {
  description = "CloudWatch retention for Lambda and worker log groups."
  type        = number
  default     = 30
  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "log_retention_days must be a CloudWatch-supported retention period."
  }
}

variable "create_log_groups" {
  description = "Create CloudWatch log groups; disable only for emulators without the Logs service."
  type        = bool
  default     = true
}
