// Security module - Key Vault, Managed Identity, and security configurations
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

@description('Virtual Network ID for private endpoints')
param vnetId string

@description('Private subnet ID for private endpoints')
param privateSubnetId string

// ========== VARIABLES ==========
var keyVaultName = 'kv-${workloadName}-${take(resourceToken, 15)}'
var userAssignedIdentityName = 'id-${workloadName}-${environmentName}-${resourceToken}'
var privateEndpointName = 'pe-kv-${workloadName}-${environmentName}-${resourceToken}'
var privateDnsZoneName = 'privatelink.vaultcore.azure.net'

// ========== USER ASSIGNED MANAGED IDENTITY ==========
resource userAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: userAssignedIdentityName
  location: location
  tags: tags
}

// ========== KEY VAULT ==========
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: tenant().tenantId
    enabledForDeployment: true
    enabledForTemplateDeployment: true
    enabledForDiskEncryption: true
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enablePurgeProtection: true
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
      virtualNetworkRules: []
      ipRules: []
    }
    publicNetworkAccess: 'Disabled'
  }
}

// ========== PRIVATE DNS ZONE FOR KEY VAULT ==========
resource privateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: privateDnsZoneName
  location: 'global'
  tags: tags
}

// ========== PRIVATE DNS ZONE LINK ==========
resource privateDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: privateDnsZone
  name: '${privateDnsZoneName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnetId
    }
  }
}

// ========== PRIVATE ENDPOINT FOR KEY VAULT ==========
resource privateEndpoint 'Microsoft.Network/privateEndpoints@2023-09-01' = {
  name: privateEndpointName
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'keyVaultConnection'
        properties: {
          privateLinkServiceId: keyVault.id
          groupIds: [
            'vault'
          ]
        }
      }
    ]
  }
}

// ========== PRIVATE DNS ZONE GROUP ==========
resource privateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-09-01' = {
  parent: privateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'keyVault'
        properties: {
          privateDnsZoneId: privateDnsZone.id
        }
      }
    ]
  }
}

// ========== ROLE ASSIGNMENTS ==========

// Key Vault Administrator role for the managed identity
resource keyVaultAdministratorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, userAssignedIdentity.id, 'Key Vault Administrator')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '00482a5a-887f-4fb3-b363-3b7fe8e74483') // Key Vault Administrator
    principalId: userAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Key Vault Secrets User role for the managed identity
resource keyVaultSecretsUserRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, userAssignedIdentity.id, 'Key Vault Secrets User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6') // Key Vault Secrets User
    principalId: userAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ========== KEY VAULT SECRETS ==========

// Database connection string secret (placeholder)
resource sqlConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'sql-connection-string'
  properties: {
    value: 'placeholder-will-be-updated-after-database-deployment'
    attributes: {
      enabled: true
    }
    contentType: 'connection-string'
  }
}

// Storage connection string secret (placeholder)
resource storageConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'storage-connection-string'
  properties: {
    value: 'placeholder-will-be-updated-after-storage-deployment'
    attributes: {
      enabled: true
    }
    contentType: 'connection-string'
  }
}

// Azure OpenAI API key secret (placeholder)
resource openAiApiKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'openai-api-key'
  properties: {
    value: 'placeholder-will-be-configured-manually'
    attributes: {
      enabled: true
    }
    contentType: 'api-key'
  }
}

// Application Insights connection string secret (placeholder)
resource appInsightsConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'appinsights-connection-string'
  properties: {
    value: 'placeholder-will-be-updated-after-monitoring-deployment'
    attributes: {
      enabled: true
    }
    contentType: 'connection-string'
  }
}

// JWT secret key for application
resource jwtSecretKey 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'jwt-secret-key'
  properties: {
    value: base64(guid(resourceGroup().id, 'jwt-secret'))
    attributes: {
      enabled: true
    }
    contentType: 'secret-key'
  }
}

// ========== OUTPUTS ==========
output keyVaultId string = keyVault.id
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri

output userAssignedIdentityId string = userAssignedIdentity.id
output userAssignedIdentityName string = userAssignedIdentity.name
output userAssignedIdentityClientId string = userAssignedIdentity.properties.clientId
output userAssignedIdentityPrincipalId string = userAssignedIdentity.properties.principalId

output privateDnsZoneId string = privateDnsZone.id
output privateEndpointId string = privateEndpoint.id

// Secret references for application configuration
output sqlConnectionStringSecretUri string = sqlConnectionStringSecret.properties.secretUri
output storageConnectionStringSecretUri string = storageConnectionStringSecret.properties.secretUri
output openAiApiKeySecretUri string = openAiApiKeySecret.properties.secretUri
output appInsightsConnectionStringSecretUri string = appInsightsConnectionStringSecret.properties.secretUri
output jwtSecretKeySecretUri string = jwtSecretKey.properties.secretUri