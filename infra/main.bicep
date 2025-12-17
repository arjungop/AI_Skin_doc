// Main Bicep template for AI Doctor Skin Lesion Application
// Deploys all required Azure services with enterprise-grade security and monitoring

targetScope = 'subscription'

@description('The name of the workload that is being deployed. Up to 10 characters long.')
@minLength(2)
@maxLength(10)
param workloadName string = 'aidoctor'

@description('The name of the environment (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environmentName string = 'dev'

@description('The location of this regional hub. All resources will be created in this region.')
param location string = deployment().location

@description('Resource naming token for unique resource names')
param resourceToken string = uniqueString(subscription().id, workloadName, environmentName)

@description('Tags to apply to all resources')
param tags object = {
  workload: workloadName
  environment: environmentName
  'azd-env-name': environmentName
  'cost-center': 'healthcare-ai'
  project: 'ai-doctor-skin-lesion'
}

// ========== RESOURCE GROUP ==========
resource resourceGroup 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: 'rg-${workloadName}-${environmentName}-${resourceToken}'
  location: location
  tags: union(tags, {
    'azd-env-name': environmentName
  })
}

// ========== NETWORKING MODULE ==========
module networking 'modules/networking.bicep' = {
  scope: resourceGroup
  name: 'networking-deployment'
  params: {
    workloadName: workloadName
    environmentName: environmentName
    location: location
    resourceToken: resourceToken
    tags: tags
  }
}

// ========== SECURITY MODULE ==========
module security 'modules/security.bicep' = {
  scope: resourceGroup
  name: 'security-deployment'
  params: {
    workloadName: workloadName
    environmentName: environmentName
    location: location
    resourceToken: resourceToken
    tags: tags
    vnetId: networking.outputs.vnetId
    privateSubnetId: networking.outputs.privateSubnetId
  }
}

// ========== STORAGE MODULE ==========
module storage 'modules/storage.bicep' = {
  scope: resourceGroup
  name: 'storage-deployment'
  params: {
    workloadName: workloadName
    environmentName: environmentName
    location: location
    resourceToken: resourceToken
    tags: tags
    vnetId: networking.outputs.vnetId
    privateSubnetId: networking.outputs.privateSubnetId
  }
}

// ========== DATABASE MODULE ==========
module database 'modules/database.bicep' = {
  scope: resourceGroup
  name: 'database-deployment'
  params: {
    workloadName: workloadName
    environmentName: environmentName
    location: location
    resourceToken: resourceToken
    tags: tags
    vnetId: networking.outputs.vnetId
    privateSubnetId: networking.outputs.privateSubnetId
    keyVaultId: security.outputs.keyVaultId
  }
}

// ========== MONITORING MODULE ==========
module monitoring 'modules/monitoring.bicep' = {
  scope: resourceGroup
  name: 'monitoring-deployment'
  params: {
    workloadName: workloadName
    environmentName: environmentName
    location: location
    resourceToken: resourceToken
    tags: tags
  }
}

// ========== COMPUTE MODULE ==========
module compute 'modules/compute.bicep' = {
  scope: resourceGroup
  name: 'compute-deployment'
  params: {
    workloadName: workloadName
    environmentName: environmentName
    location: location
    resourceToken: resourceToken
    tags: tags
    vnetId: networking.outputs.vnetId
    appSubnetId: networking.outputs.appSubnetId
    privateSubnetId: networking.outputs.privateSubnetId
    keyVaultId: security.outputs.keyVaultId
    sqlConnectionString: database.outputs.sqlConnectionString
    storageConnectionString: storage.outputs.storageConnectionString
    applicationInsightsConnectionString: monitoring.outputs.applicationInsightsConnectionString
    logAnalyticsWorkspaceId: monitoring.outputs.logAnalyticsWorkspaceId
    userAssignedIdentityId: security.outputs.userAssignedIdentityId
    userAssignedIdentityClientId: security.outputs.userAssignedIdentityClientId
  }
}

// ========== AZURE AD B2C MODULE ==========
module authentication 'modules/authentication.bicep' = {
  scope: resourceGroup
  name: 'authentication-deployment'
  params: {
    workloadName: workloadName
    environmentName: environmentName
    location: location
    resourceToken: resourceToken
    tags: tags
    webAppUrl: compute.outputs.webAppUrl
    apiUrl: compute.outputs.apiUrl
  }
}

// ========== BACKUP & RECOVERY MODULE ==========
module backup 'modules/backup.bicep' = {
  scope: resourceGroup
  name: 'backup-deployment'
  params: {
    workloadName: workloadName
    environmentName: environmentName
    location: location
    resourceToken: resourceToken
    tags: tags
    sqlServerId: database.outputs.sqlServerId
    storageAccountId: storage.outputs.storageAccountId
    appServiceId: compute.outputs.appServiceId
  }
}

// ========== OUTPUTS ==========
output resourceGroupName string = resourceGroup.name
output location string = location
output environmentName string = environmentName

// Networking outputs
output vnetId string = networking.outputs.vnetId
output vnetName string = networking.outputs.vnetName
output applicationGatewayFqdn string = networking.outputs.applicationGatewayFqdn

// Security outputs
output keyVaultName string = security.outputs.keyVaultName
output keyVaultUri string = security.outputs.keyVaultUri
output userAssignedIdentityName string = security.outputs.userAssignedIdentityName

// Storage outputs
output storageAccountName string = storage.outputs.storageAccountName
output blobContainerName string = storage.outputs.blobContainerName

// Database outputs
output sqlServerName string = database.outputs.sqlServerName
output sqlDatabaseName string = database.outputs.sqlDatabaseName

// Compute outputs
output appServiceName string = compute.outputs.appServiceName
output webAppUrl string = compute.outputs.webAppUrl
output apiUrl string = compute.outputs.apiUrl
output functionsAppName string = compute.outputs.functionsAppName
output functionsUrl string = compute.outputs.functionsUrl
output containerAppsEnvironmentName string = compute.outputs.containerAppsEnvironmentName

// Monitoring outputs
output logAnalyticsWorkspaceName string = monitoring.outputs.logAnalyticsWorkspaceName
output applicationInsightsName string = monitoring.outputs.applicationInsightsName

// Authentication outputs
output azureAdB2cTenant string = authentication.outputs.azureAdB2cTenant
output azureAdClientId string = authentication.outputs.azureAdClientId

// Backup outputs
output recoveryServicesVaultName string = backup.outputs.recoveryServicesVaultName
output backupVaultName string = backup.outputs.backupVaultName