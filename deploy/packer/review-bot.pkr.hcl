packer {
  required_plugins {
    amazon = {
      source  = "github.com/hashicorp/amazon"
      version = "= 1.3.9"
    }
    lxd = {
      source  = "github.com/hashicorp/lxd"
      version = "= 1.0.3"
    }
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "subnet_id" {
  type    = string
  default = null
}

variable "ssh_source_cidr" {
  type    = string
  default = "127.0.0.1/32"
}

variable "instance_type" {
  type    = string
  default = "t3.large"
}

variable "repo_source_path" {
  type    = string
  default = "."
}

variable "review_bot_wheel_dir" {
  type    = string
  default = "deploy/terraform/dist/wheel/"
}

variable "lxd_source_image" {
  type    = string
  default = "ubuntu:24.04"
}

variable "lxd_output_image" {
  type    = string
  default = "review-bot-worker"
}

# Seconds to wait after the build VM starts: file/exec provisioners fail until lxd-agent runs (~20s).
variable "lxd_init_sleep_seconds" {
  type    = string
  default = "90"
}

variable "lxd_profile" {
  type    = string
  default = "review-bot-worker"
}

source "amazon-ebs" "review_bot" {
  region        = var.region
  instance_type = var.instance_type
  subnet_id     = var.subnet_id
  ssh_username  = "ubuntu"
  ami_name      = "review-bot-ubuntu-24-04-${formatdate("YYYYMMDDhhmmss", timestamp())}"

  source_ami_filter {
    filters = {
      name                = "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["099720109477"]
  }

  temporary_security_group_source_cidrs = [var.ssh_source_cidr]

  ami_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 40
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }
}

source "lxd" "review_bot" {
  container_name  = "packer-review-bot"
  init_sleep      = var.lxd_init_sleep_seconds
  image           = var.lxd_source_image
  output_image    = var.lxd_output_image
  profile         = var.lxd_profile
  virtual_machine = true

  launch_config = {
    "limits.cpu"    = "4"
    "limits.memory" = "8GiB"
  }
}

build {
  sources = ["source.amazon-ebs.review_bot", "source.lxd.review_bot"]

  provisioner "file" {
    source      = var.repo_source_path
    destination = "/tmp/review-bot-source/"
  }

  provisioner "shell" {
    inline = ["mkdir -p /tmp/review-bot-wheel"]
  }

  provisioner "file" {
    source      = "${trimsuffix(var.review_bot_wheel_dir, "/")}/"
    destination = "/tmp/review-bot-wheel/"
  }

  provisioner "file" {
    source      = "deploy/packer/bootstrap"
    destination = "/tmp/review-bot-bootstrap"
  }

  provisioner "file" {
    source      = "deploy/packer/bootstrap_runtime.py"
    destination = "/tmp/review-bot-bootstrap-runtime.py"
  }

  provisioner "file" {
    source      = "deploy/packer/review-bot-bootstrap.service"
    destination = "/tmp/review-bot-bootstrap.service"
  }

  provisioner "shell" {
    execute_command = "sudo -E -S sh '{{ .Path }}'"
    inline = [
      "set -eu",
      "export DEBIAN_FRONTEND=noninteractive",
      "install -m 0755 -d /etc/apt/keyrings",
      "curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc",
      "chmod a+r /etc/apt/keyrings/docker.asc",
      ". /etc/os-release && echo \"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable\" > /etc/apt/sources.list.d/docker.list",
      "apt-get update",
      "apt-get install -y --no-install-recommends ca-certificates curl dbus-user-session docker-ce docker-ce-cli docker-ce-rootless-extras containerd.io fuse-overlayfs openjdk-21-jdk-headless python3 python3-pip python3-venv slirp4netns uidmap",
      "systemctl disable --now docker.socket docker.service",
      "systemctl stop docker.socket docker.service",
      "if systemctl is-active --quiet docker.socket || systemctl is-active --quiet docker.service; then echo 'rootful Docker still active' >&2; exit 1; fi",
      "useradd --uid 1001 --create-home --shell /bin/bash --groups docker reviewbot",
      "grep -q '^reviewbot:' /etc/subuid || echo 'reviewbot:100000:65536' >> /etc/subuid",
      "grep -q '^reviewbot:' /etc/subgid || echo 'reviewbot:100000:65536' >> /etc/subgid",
      "install -d -m 0755 /etc/systemd/system/user@1001.service.d",
      "printf '[Service]\\nDelegate=yes\\n' > /etc/systemd/system/user@1001.service.d/delegate.conf",
      "systemctl daemon-reload",
      "loginctl enable-linger reviewbot",
      "systemctl start user@1001.service",
      "runuser -u reviewbot -- env HOME=/home/reviewbot USER=reviewbot LOGNAME=reviewbot XDG_RUNTIME_DIR=/run/user/1001 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus dockerd-rootless-setuptool.sh install",
      "runuser -u reviewbot -- env HOME=/home/reviewbot USER=reviewbot LOGNAME=reviewbot XDG_RUNTIME_DIR=/run/user/1001 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus systemctl --user enable --now docker",
      "systemctl enable --now docker.service",
      "install -d -o root -g root -m 0755 /opt/review-bot/bin /etc/review-bot /var/lib/review-bot",
      "python3 -m venv /opt/review-bot/venv",
      "/opt/review-bot/venv/bin/pip install --no-cache-dir /tmp/review-bot-wheel/*.whl boto3",
      "install -m 0755 /tmp/review-bot-bootstrap /opt/review-bot/bin/bootstrap",
      "install -m 0644 /tmp/review-bot-bootstrap-runtime.py /opt/review-bot/bootstrap_runtime.py",
      "ln -s /opt/review-bot/venv/bin/review-bot-ondemand /opt/review-bot/bin/review-bot-ondemand",
      "install -m 0644 /tmp/review-bot-bootstrap.service /etc/systemd/system/review-bot-bootstrap.service",
      "cd /tmp/review-bot-source",
      "rm -rf dist",
      "mkdir -p dist",
      "cp /tmp/review-bot-wheel/*.whl dist/",
      "python3 tooling/fetch_tools.py && python3 tooling/verify_tools.py",
      "runuser -u reviewbot -- env HOME=/home/reviewbot USER=reviewbot LOGNAME=reviewbot docker build --tag review-bot-application:local .",
      "runuser -u reviewbot -- env HOME=/home/reviewbot USER=reviewbot LOGNAME=reviewbot docker pull amazoncorretto:21",
      "runuser -u reviewbot -- env HOME=/home/reviewbot USER=reviewbot LOGNAME=reviewbot docker build --build-arg REVIEW_BOT_IMAGE=review-bot-application:local --build-arg JDK_IMAGE=amazoncorretto:21 --file Dockerfile.pmd --tag review-bot-pmd:7.3.0 .",
      "runuser -u reviewbot -- env HOME=/home/reviewbot USER=reviewbot LOGNAME=reviewbot docker build --file deploy/agent-sandbox/Dockerfile.agent --tag review-bot-pi-agent:0.73.1 .",
      "runuser -u reviewbot -- env HOME=/home/reviewbot USER=reviewbot LOGNAME=reviewbot docker build --file deploy/agent-sandbox/Dockerfile.proxy --tag \"$(/opt/review-bot/venv/bin/review-bot agent-proxy-image)\" deploy/agent-sandbox",
      "runuser -u reviewbot -- env HOME=/home/reviewbot USER=reviewbot LOGNAME=reviewbot docker save --output /tmp/review-bot-pmd.tar review-bot-pmd:7.3.0",
      "runuser -u reviewbot -- env HOME=/home/reviewbot USER=reviewbot LOGNAME=reviewbot XDG_RUNTIME_DIR=/run/user/1001 DOCKER_HOST=unix:///run/user/1001/docker.sock docker load --input /tmp/review-bot-pmd.tar",
      "runuser -u reviewbot -- env HOME=/home/reviewbot USER=reviewbot LOGNAME=reviewbot XDG_RUNTIME_DIR=/run/user/1001 DOCKER_HOST=unix:///run/user/1001/docker.sock docker image inspect --format '{{.Id}}' review-bot-pmd:7.3.0 > /etc/review-bot/pmd-image-pin",
      "systemctl daemon-reload",
      "systemctl enable review-bot-bootstrap.service",
      "java -version",
      "python3 --version",
      "rm -rf /tmp/review-bot-source /tmp/review-bot-wheel /tmp/review-bot-pmd.tar /tmp/review-bot-bootstrap /tmp/review-bot-bootstrap-runtime.py /tmp/review-bot-bootstrap.service"
    ]
  }
}
