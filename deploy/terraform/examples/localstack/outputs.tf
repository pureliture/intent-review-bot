output "webhook_url" {
  value = module.on_demand.webhook_url
}

output "job_table_name" {
  value = module.on_demand.job_table_name
}

output "session_bucket" {
  value = module.on_demand.session_bucket
}

output "intake_function_name" {
  value = module.on_demand.intake_function_name
}

output "recovery_function_name" {
  value = module.on_demand.recovery_function_name
}

output "launch_template_id" {
  value = module.on_demand.launch_template_id
}
