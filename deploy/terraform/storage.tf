resource "aws_dynamodb_table" "jobs" {
  #checkov:skip=CKV_AWS_119:DynamoDB uses AWS-owned encryption by default; a customer-managed KMS key is not required and would add cost.
  name         = "${var.name_prefix}-jobs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"

  attribute {
    name = "pk"
    type = "S"
  }
  attribute {
    name = "pr_key"
    type = "S"
  }
  attribute {
    name = "active"
    type = "S"
  }

  global_secondary_index {
    name            = "pr_index"
    hash_key        = "pr_key"
    projection_type = "KEYS_ONLY"
  }
  global_secondary_index {
    name            = "active_index"
    hash_key        = "active"
    projection_type = "KEYS_ONLY"
  }

  point_in_time_recovery {
    enabled = true
  }
  server_side_encryption {
    enabled = true
  }

  tags = {
    Name = "${var.name_prefix}-jobs"
  }
}

resource "aws_secretsmanager_secret" "webhook" {
  #checkov:skip=CKV_AWS_149:Secrets Manager uses its AWS-managed encryption key; no customer-managed KMS key is required.
  #checkov:skip=CKV2_AWS_57:Webhook secret rotation is operator-managed and GitHub must be updated in lockstep.
  name                    = "${var.name_prefix}/webhook-secret"
  description             = "GitHub App webhook HMAC secret; set its value out of band."
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret" "github_app" {
  #checkov:skip=CKV_AWS_149:Secrets Manager uses its AWS-managed encryption key; no customer-managed KMS key is required.
  #checkov:skip=CKV2_AWS_57:GitHub App credential rotation is operator-managed and must be coordinated with GitHub.
  name                    = "${var.name_prefix}/github-app"
  description             = "JSON containing app_id and private_key; set its value out of band."
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret" "openrouter" {
  #checkov:skip=CKV_AWS_149:Secrets Manager uses its AWS-managed encryption key; no customer-managed KMS key is required.
  #checkov:skip=CKV2_AWS_57:OpenRouter API key rotation is operator-managed by the account owner.
  name                    = "${var.name_prefix}/openrouter"
  description             = "OpenRouter API key; set its value out of band."
  recovery_window_in_days = 0
}

resource "aws_s3_bucket" "sessions" {
  #checkov:skip=CKV_AWS_144:Cross-region replication is not requested and would add another bucket and transfer costs.
  #checkov:skip=CKV_AWS_21:Versioning is intentionally suspended as explicitly required for short-lived session artifacts.
  #checkov:skip=CKV2_AWS_62:No event consumer is part of the session-artifact contract.
  #checkov:skip=CKV_AWS_145:SSE-S3 (AES256) is enabled as requested; a customer-managed KMS key is unnecessary and metered.
  #checkov:skip=CKV_AWS_18:Access logging would require a separate log bucket and is outside the requested minimal artifact store.
  bucket_prefix = "${var.name_prefix}-sessions-"

  tags = {
    Name = "${var.name_prefix}-sessions"
  }
}

resource "aws_s3_bucket_public_access_block" "sessions" {
  bucket                  = aws_s3_bucket.sessions.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "sessions" {
  bucket = aws_s3_bucket.sessions.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

#checkov:skip=CKV_AWS_21:Session objects are deliberately short-lived and the requested bucket must have versioning off.
resource "aws_s3_bucket_versioning" "sessions" {
  bucket = aws_s3_bucket.sessions.id
  versioning_configuration {
    status = "Suspended"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "sessions" {
  bucket = aws_s3_bucket.sessions.id
  rule {
    id     = "expire-session-artifacts"
    status = "Enabled"
    filter {}
    expiration {
      days = var.session_expiration_days
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_policy" "sessions_tls_only" {
  bucket = aws_s3_bucket.sessions.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.sessions.arn, "${aws_s3_bucket.sessions.arn}/*"]
      Condition = {
        Bool = {
          "aws:SecureTransport" = "false"
        }
      }
    }]
  })
}

resource "aws_cloudwatch_log_group" "intake" {
  #checkov:skip=CKV_AWS_158:CloudWatch Logs uses its AWS-managed encryption by default; a customer KMS key is not required.
  #checkov:skip=CKV_AWS_338:Retention is configurable and defaults to 30 days to bound storage cost for this on-demand service.
  count             = var.create_log_groups ? 1 : 0
  name              = "/aws/lambda/${local.lambda_intake_name}"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "recovery" {
  #checkov:skip=CKV_AWS_158:CloudWatch Logs uses its AWS-managed encryption by default; a customer KMS key is not required.
  #checkov:skip=CKV_AWS_338:Retention is configurable and defaults to 30 days to bound storage cost for this on-demand service.
  count             = var.create_log_groups ? 1 : 0
  name              = "/aws/lambda/${local.lambda_recovery_name}"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "worker" {
  #checkov:skip=CKV_AWS_158:CloudWatch Logs uses its AWS-managed encryption by default; a customer KMS key is not required.
  #checkov:skip=CKV_AWS_338:Retention is configurable and defaults to 30 days to bound storage cost for this on-demand service.
  count             = var.create_log_groups ? 1 : 0
  name              = local.worker_log_group
  retention_in_days = var.log_retention_days
}
