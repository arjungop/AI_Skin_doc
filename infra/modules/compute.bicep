// Compute module - App Service, Azure Functions, Container Apps
@description('The name of the workload')
param workloadName string
@description('The name of the environment')
param environmentName string
@description('The Azure region where resources will be created')
param location string
@description('Resource naming token')
param resourceToken string
@description('Tags to apply to resources')
param tags object
@description('Virtual Network ID')
param vnetId string
@description('App subnet ID')
param appSubnetId string
@description('Private subnet ID')
param privateSubnetId string
@description('Key Vault ID')
param keyVaultId string
@description('SQL connection string')
param sqlConnectionString string
@description('Storage connection string')
param storageConnectionString string
@description('Application Insights connection string')
param applicationInsightsConnectionString string
@description('Log Analytics workspace ID')
param logAnalyticsWorkspaceId string
@description('User assigned identity ID')
param userAssignedIdentityId string
@description('User assigned identity client ID')
param userAssignedIdentityClientId string

// Variables
var appServicePlanName = 'asp-${workloadName}-${environmentName}-${resourceToken}'
var appServiceName = 'app-${workloadName}-${environmentName}-${resourceToken}'
var functionsAppName = 'func-${workloadName}-${environmentName}-${resourceToken}'
var staticWebAppName = 'stapp-${workloadName}-${environmentName}-${resourceToken}'
var containerAppsEnvironmentName = 'cae-${workloadName}-${environmentName}-${resourceToken}'

// App Service Plan
resource appServicePlan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: appServicePlanName
  location: location
  tags: tags
  sku: {
    name: 'P1v3'
    tier: 'PremiumV3'
    size: 'P1v3'
    family: 'Pv3'
    capacity: 1
  }
  properties: {
    reserved: true
  }
  kind: 'linux'
}

// App Service
resource appService 'Microsoft.Web/sites@2023-12-01' = {
  name: appServiceName
  location: location
  tags: union(tags, {'azd-service-name': 'api'})
  kind: 'app,linux,container'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userAssignedIdentityId}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    virtualNetworkSubnetId: appSubnetId
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'DOCKER|python:3.11-slim'
      alwaysOn: true
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      appSettings: [
        { name: 'AZURE_SQL_CONNECTION_STRING', value: '@Microsoft.KeyVault(SecretUri=${keyVaultId}/secrets/sql-connection-string/)' }
        { name: 'AZURE_STORAGE_CONNECTION_STRING', value: '@Microsoft.KeyVault(SecretUri=${keyVaultId}/secrets/storage-connection-string/)' }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: applicationInsightsConnectionString }
        { name: 'AZURE_CLIENT_ID', value: userAssignedIdentityClientId }
      ]
      cors: {
        allowedOrigins: ['*']
        supportCredentials: false
      }
    }
  }
}

// Functions App
resource functionsApp 'Microsoft.Web/sites@2023-12-01' = {
  name: functionsAppName
  location: location
  tags: union(tags, {'azd-service-name': 'functions'})
  kind: 'functionapp,linux'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userAssignedIdentityId}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    virtualNetworkSubnetId: appSubnetId
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'Python|3.11'
      alwaysOn: false
      appSettings: [
        { name: 'AzureWebJobsStorage', value: storageConnectionString }
        { name: 'FUNCTIONS_EXTENSION_VERSION', value: '~4' }
        { name: 'FUNCTIONS_WORKER_RUNTIME', value: 'python' }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: applicationInsightsConnectionString }
        { name: 'AZURE_CLIENT_ID', value: userAssignedIdentityClientId }
      ]
    }
  }
}

// Static Web App
resource staticWebApp 'Microsoft.Web/staticSites@2023-12-01' = {
  name: staticWebAppName
  location: location
  tags: union(tags, {'azd-service-name': 'web'})
  sku: {
    name: 'Standard'
    tier: 'Standard'
  }
  properties: {
    buildProperties: {
      skipGithubActionWorkflowGeneration: true
    }
  }
}

// Container Apps Environment
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerAppsEnvironmentName
  location: location
  tags: tags
  properties: {
    vnetConfiguration: {
      infrastructureSubnetId: privateSubnetId
    }
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: reference(logAnalyticsWorkspaceId, '2023-09-01').customerId
        sharedKey: listKeys(logAnalyticsWorkspaceId, '2023-09-01').primarySharedKey
      }
    }
  }
}

// Outputs
output appServicePlanId string = appServicePlan.id
output appServiceId string = appService.id
output appServiceName string = appService.name
output webAppUrl string = 'https://${appService.properties.defaultHostName}'
output apiUrl string = 'https://${appService.properties.defaultHostName}'
output functionsAppId string = functionsApp.id
output functionsAppName string = functionsApp.name
output functionsUrl string = 'https://${functionsApp.properties.defaultHostName}'
output staticWebAppId string = staticWebApp.id
output staticWebAppName string = staticWebApp.name
output staticWebAppUrl string = 'https://${staticWebApp.properties.defaultHostname}'
output containerAppsEnvironmentId string = containerAppsEnvironment.id
output containerAppsEnvironmentName string = containerAppsEnvironment.name