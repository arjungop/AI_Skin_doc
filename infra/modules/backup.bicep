// Backup module - Azure Backup and Site Recovery
param workloadName string
param environmentName string
param location string
param resourceToken string
param tags object
param sqlServerId string
param storageAccountId string
param appServiceId string

var recoveryServicesVaultName = 'rsv-${workloadName}-${environmentName}-${resourceToken}'
var backupVaultName = 'bv-${workloadName}-${environmentName}-${resourceToken}'

// Recovery Services Vault
resource recoveryServicesVault 'Microsoft.RecoveryServices/vaults@2024-04-01' = {
  name: recoveryServicesVaultName
  location: location
  tags: tags
  sku: {
    name: 'Standard'
    tier: 'Standard'
  }
  properties: {}
}

// Backup Vault for modern workloads
resource backupVault 'Microsoft.DataProtection/backupVaults@2024-04-01' = {
  name: backupVaultName
  location: location
  tags: tags
  properties: {
    storageSettings: [{
      datastoreType: 'VaultStore'
      type: 'GeoRedundant'
    }]
  }
  identity: {
    type: 'SystemAssigned'
  }
}

output recoveryServicesVaultId string = recoveryServicesVault.id
output recoveryServicesVaultName string = recoveryServicesVault.name
output backupVaultId string = backupVault.id
output backupVaultName string = backupVault.name