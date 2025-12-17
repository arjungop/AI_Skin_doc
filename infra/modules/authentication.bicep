// Authentication module - Azure AD B2C
param workloadName string
param environmentName string
param location string
param resourceToken string
param tags object
param webAppUrl string
param apiUrl string

var b2cTenantName = '${workloadName}${environmentName}${take(resourceToken, 6)}'

// Azure AD B2C will be created manually via Azure Portal
// This module provides the configuration template

output azureAdB2cTenant string = '${b2cTenantName}.onmicrosoft.com'
output azureAdClientId string = 'will-be-configured-after-b2c-setup'
output azureAdTenantId string = 'will-be-configured-after-b2c-setup'