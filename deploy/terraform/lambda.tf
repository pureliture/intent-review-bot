resource "aws_lambda_function" "intake" {
  #checkov:skip=CKV_AWS_173:Environment values are identifiers and non-secret configuration; webhook and app secrets are fetched by ARN at runtime.
  #checkov:skip=CKV_AWS_50:X-Ray is not required by the request and would add metered tracing to the free deployment profile.
  #checkov:skip=CKV_AWS_117:The intake Function URL is public by design; no Lambda VPC attachment is required for the webhook contract.
  #checkov:skip=CKV_AWS_116:Job metadata and the scheduled recovery Lambda provide the required retry/recovery path without a separate queue.
  #checkov:skip=CKV_AWS_272:Lambda ZIPs are operator-built and supplied by path; code signing is outside this deployment contract.
  function_name    = local.lambda_intake_name
  role             = aws_iam_role.lambda.arn
  runtime          = "python3.11"
  handler          = "review_bot.runtime.aws_lambda.intake"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)
  timeout          = 20
  memory_size      = 256
  architectures    = ["x86_64"]

  reserved_concurrent_executions = 10

  environment {
    variables = {
      REVIEW_BOT_JOB_TABLE          = aws_dynamodb_table.jobs.name
      REVIEW_BOT_LAUNCH_TEMPLATE_ID = aws_launch_template.worker.id
      REVIEW_BOT_WEBHOOK_SECRET_ARN = aws_secretsmanager_secret.webhook.arn
      REVIEW_BOT_INSTANCE_TAGS = jsonencode(merge({
        "review-bot:job-table"             = aws_dynamodb_table.jobs.name
        "review-bot:github-app-secret-arn" = aws_secretsmanager_secret.github_app.arn
        "review-bot:openrouter-secret-arn" = aws_secretsmanager_secret.openrouter.arn
        "review-bot:session-bucket"        = aws_s3_bucket.sessions.bucket
        "review-bot:region"                = data.aws_region.current.name
      }, local.agent_instance_tags))
      REVIEW_BOT_ALLOWED_INSTALLATIONS = join(",", var.allowed_installations)
      REVIEW_BOT_ALLOWED_REPOSITORIES  = join(",", var.allowed_repositories)
      REVIEW_BOT_MAX_VM_SECONDS        = tostring(var.max_vm_seconds)
    }
  }

  depends_on = [aws_cloudwatch_log_group.intake, aws_iam_role_policy.lambda]
}

resource "aws_lambda_function" "recovery" {
  #checkov:skip=CKV_AWS_173:Environment values are identifiers and non-secret configuration; no secret value is stored here.
  #checkov:skip=CKV_AWS_50:X-Ray is not required by the request and would add metered tracing to the free deployment profile.
  #checkov:skip=CKV_AWS_117:Recovery requires only AWS control-plane APIs; a Lambda VPC attachment is not required by this contract.
  #checkov:skip=CKV_AWS_116:Recovery is invoked every five minutes by EventBridge and uses durable DynamoDB job state.
  #checkov:skip=CKV_AWS_272:Lambda ZIPs are operator-built and supplied by path; code signing is outside this deployment contract.
  function_name    = local.lambda_recovery_name
  role             = aws_iam_role.lambda.arn
  runtime          = "python3.11"
  handler          = "review_bot.runtime.aws_lambda.recovery"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)
  timeout          = 60
  memory_size      = 256
  architectures    = ["x86_64"]

  reserved_concurrent_executions = 1

  environment {
    variables = {
      REVIEW_BOT_JOB_TABLE          = aws_dynamodb_table.jobs.name
      REVIEW_BOT_LAUNCH_TEMPLATE_ID = aws_launch_template.worker.id
      REVIEW_BOT_INSTANCE_TAGS = jsonencode(merge({
        "review-bot:job-table"             = aws_dynamodb_table.jobs.name
        "review-bot:github-app-secret-arn" = aws_secretsmanager_secret.github_app.arn
        "review-bot:openrouter-secret-arn" = aws_secretsmanager_secret.openrouter.arn
        "review-bot:session-bucket"        = aws_s3_bucket.sessions.bucket
        "review-bot:region"                = data.aws_region.current.name
      }, local.agent_instance_tags))
      REVIEW_BOT_MAX_VM_SECONDS = tostring(var.max_vm_seconds)
    }
  }

  depends_on = [aws_cloudwatch_log_group.recovery, aws_iam_role_policy.lambda]
}

resource "aws_lambda_function_url" "intake" {
  #checkov:skip=CKV_AWS_258:GitHub cannot sign AWS_IAM requests; the public URL is required and the handler validates GitHub's HMAC-SHA256 signature before processing.
  function_name      = aws_lambda_function.intake.function_name
  authorization_type = "NONE"
}

resource "aws_lambda_permission" "intake_url" {
  #checkov:skip=CKV_AWS_301:Public invocation is required for GitHub delivery; the only public entry is the Function URL with the handler's HMAC validation.
  statement_id           = "AllowPublicFunctionUrl"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.intake.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

resource "aws_cloudwatch_event_rule" "recovery" {
  name                = "${var.name_prefix}-recovery"
  description         = "Reconcile lost jobs and managed worker instances."
  schedule_expression = "rate(5 minutes)"
}

resource "aws_cloudwatch_event_target" "recovery" {
  rule      = aws_cloudwatch_event_rule.recovery.name
  target_id = "recovery"
  arn       = aws_lambda_function.recovery.arn
}

resource "aws_lambda_permission" "recovery_schedule" {
  statement_id  = "AllowEventBridgeRecovery"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.recovery.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.recovery.arn
}
