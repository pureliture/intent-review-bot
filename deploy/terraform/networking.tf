#checkov:skip=CKV_AWS_382:Workers need outbound HTTPS to GitHub and OpenRouter; ingress is absent and no other egress is allowed.
resource "aws_security_group" "worker" {
  name                   = "${var.name_prefix}-worker"
  description            = "One-shot review workers: no ingress, outbound HTTPS only."
  vpc_id                 = var.vpc_id
  revoke_rules_on_delete = true

  egress {
    description = "HTTPS to GitHub, OpenRouter, and AWS APIs"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name_prefix}-worker"
  }
}

resource "aws_launch_template" "worker" {
  name                                 = "${var.name_prefix}-worker"
  image_id                             = var.worker_ami_id
  instance_type                        = var.worker_instance_type
  instance_initiated_shutdown_behavior = "terminate"

  iam_instance_profile {
    arn = aws_iam_instance_profile.worker.arn
  }

  network_interfaces {
    device_index                = 0
    subnet_id                   = var.subnet_id
    security_groups             = [aws_security_group.worker.id]
    associate_public_ip_address = false
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      encrypted             = true
      volume_type           = "gp3"
      volume_size           = 40
      delete_on_termination = true
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      "review-bot:managed"               = "true"
      "review-bot:job-table"             = aws_dynamodb_table.jobs.name
      "review-bot:github-app-secret-arn" = aws_secretsmanager_secret.github_app.arn
      "review-bot:openrouter-secret-arn" = aws_secretsmanager_secret.openrouter.arn
      "review-bot:session-bucket"        = aws_s3_bucket.sessions.bucket
      "review-bot:region"                = data.aws_region.current.name
    }
  }
  tag_specifications {
    resource_type = "volume"
    tags = {
      "review-bot:managed" = "true"
    }
  }

  tags = {
    Name = "${var.name_prefix}-worker"
  }
}
