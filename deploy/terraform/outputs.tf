output "webhook_url" {
  description = "GitHub App webhook endpoint; register this URL after apply."
  value       = aws_lambda_function_url.intake.function_url
}

output "job_table_name" {
  value = aws_dynamodb_table.jobs.name
}

output "session_bucket" {
  value = aws_s3_bucket.sessions.bucket
}

output "webhook_secret_arn" {
  value = aws_secretsmanager_secret.webhook.arn
}

output "github_app_secret_arn" {
  value = aws_secretsmanager_secret.github_app.arn
}

output "openrouter_secret_arn" {
  value = aws_secretsmanager_secret.openrouter.arn
}

output "launch_template_id" {
  value = aws_launch_template.worker.id
}

output "intake_function_name" {
  value = aws_lambda_function.intake.function_name
}

output "recovery_function_name" {
  value = aws_lambda_function.recovery.function_name
}

output "worker_role_arn" {
  value = aws_iam_role.worker.arn
}
