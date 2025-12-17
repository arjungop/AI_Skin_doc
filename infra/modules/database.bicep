// Database module - Azure SQL Database with private endpoints and security
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

@description('Key Vault ID for storing secrets')
param keyVaultId string

@secure()
@description('SQL Server administrator password')
param sqlAdminPassword string = newGuid()

// ========== VARIABLES ==========
var sqlServerName = 'sql-${workloadName}-${environmentName}-${resourceToken}'
var sqlDatabaseName = 'db-${workloadName}-${environmentName}'
var privateEndpointName = 'pe-sql-${workloadName}-${environmentName}-${resourceToken}'
var privateDnsZoneName = 'privatelink${environment().suffixes.sqlServerHostname}'
var sqlAdminLogin = 'aidoctoradmin'

// ========== SQL SERVER ==========
resource sqlServer 'Microsoft.Sql/servers@2023-05-01-preview' = {
  name: sqlServerName
  location: location
  tags: tags
  properties: {
    administratorLogin: sqlAdminLogin
    administratorLoginPassword: sqlAdminPassword
    version: '12.0'
    minimalTlsVersion: '1.2'
    publicNetworkAccess: 'Disabled'
    restrictOutboundNetworkAccess: 'Disabled'
  }
  identity: {
    type: 'SystemAssigned'
  }
}

// ========== SQL DATABASE ==========
resource sqlDatabase 'Microsoft.Sql/servers/databases@2023-05-01-preview' = {
  parent: sqlServer
  name: sqlDatabaseName
  location: location
  tags: tags
  sku: {
    name: 'GP_S_Gen5'
    tier: 'GeneralPurpose'
    family: 'Gen5'
    capacity: 1
  }
  properties: {
    collation: 'SQL_Latin1_General_CP1_CI_AS'
    maxSizeBytes: 34359738368 // 32 GB
    catalogCollation: 'SQL_Latin1_General_CP1_CI_AS'
    zoneRedundant: false
    readScale: 'Disabled'
    requestedBackupStorageRedundancy: 'Geo'
    isLedgerOn: false
    autoPauseDelay: 60 // Auto-pause after 1 hour for cost optimization
    minCapacity: json('0.5')
  }
}

// ========== SQL SERVER FIREWALL RULES ==========
// Allow Azure services (for backup and monitoring)
resource firewallRuleAzureServices 'Microsoft.Sql/servers/firewallRules@2023-05-01-preview' = {
  parent: sqlServer
  name: 'AllowAllWindowsAzureIps'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// ========== PRIVATE DNS ZONE ==========
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

// ========== PRIVATE ENDPOINT ==========
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
        name: 'sqlConnection'
        properties: {
          privateLinkServiceId: sqlServer.id
          groupIds: [
            'sqlServer'
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
        name: 'sqlServer'
        properties: {
          privateDnsZoneId: privateDnsZone.id
        }
      }
    ]
  }
}

// ========== AUDITING CONFIGURATION ==========
resource sqlServerAuditing 'Microsoft.Sql/servers/auditingSettings@2023-05-01-preview' = {
  parent: sqlServer
  name: 'default'
  properties: {
    state: 'Enabled'
    retentionDays: 90
    isAzureMonitorTargetEnabled: true
  }
}

// ========== SECURITY ALERT POLICY ==========
resource sqlSecurityAlertPolicy 'Microsoft.Sql/servers/securityAlertPolicies@2023-05-01-preview' = {
  parent: sqlServer
  name: 'default'
  properties: {
    state: 'Enabled'
    disabledAlerts: []
    emailAddresses: []
    emailAccountAdmins: true
    retentionDays: 90
  }
}

// ========== VULNERABILITY ASSESSMENT ==========
// Note: Vulnerability assessment requires storage container configuration
// This will be configured post-deployment with proper storage container

// ========== TRANSPARENT DATA ENCRYPTION ==========
resource sqlDatabaseTDE 'Microsoft.Sql/servers/databases/transparentDataEncryption@2023-05-01-preview' = {
  parent: sqlDatabase
  name: 'current'
  properties: {
    state: 'Enabled'
  }
}

// ========== SHORT TERM BACKUP RETENTION ==========
resource sqlDatabaseBackupShortTermRetention 'Microsoft.Sql/servers/databases/backupShortTermRetentionPolicies@2023-05-01-preview' = {
  parent: sqlDatabase
  name: 'default'
  properties: {
    retentionDays: 35
    diffBackupIntervalInHours: 12
  }
}

// ========== LONG TERM BACKUP RETENTION ==========
resource sqlDatabaseBackupLongTermRetention 'Microsoft.Sql/servers/databases/backupLongTermRetentionPolicies@2023-05-01-preview' = {
  parent: sqlDatabase
  name: 'default'
  properties: {
    weeklyRetention: 'P12W'   // 12 weeks
    monthlyRetention: 'P12M'  // 12 months
    yearlyRetention: 'P7Y'    // 7 years
    weekOfYear: 1
  }
}

// ========== OUTPUTS ==========
output sqlServerId string = sqlServer.id
output sqlServerName string = sqlServer.name
output sqlServerFqdn string = sqlServer.properties.fullyQualifiedDomainName
output sqlDatabaseId string = sqlDatabase.id
output sqlDatabaseName string = sqlDatabase.name

// Connection string for applications (will be stored in Key Vault)
output sqlConnectionString string = 'Server=tcp:${sqlServer.properties.fullyQualifiedDomainName},1433;Initial Catalog=${sqlDatabase.name};Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;Authentication=Active Directory Managed Identity;'

output privateEndpointId string = privateEndpoint.id
output privateDnsZoneId string = privateDnsZone.id