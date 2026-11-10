terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

data "aws_region" "current" {}

data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

locals {
  lambda_intake_name   = "${var.name_prefix}-intake"
  lambda_recovery_name = "${var.name_prefix}-recovery"
  worker_log_group     = "/${var.name_prefix}/worker"
  lambda_log_group_arns = [
    "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${local.lambda_intake_name}:*",
    "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${local.lambda_recovery_name}:*",
  ]
  worker_log_group_arn = "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:${local.worker_log_group}:*"
  agent_instance_tags = merge(
    {
      "review-bot:agent-provider" = var.agent_provider
      "review-bot:agent-model-id" = var.agent_model_id
    },
    var.agent_context_window_tokens == null ? {} : {
      "review-bot:agent-context-window-tokens" = tostring(var.agent_context_window_tokens)
    },
  )
}
