// Storage module - Azure Blob Storage with private endpoints and security
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
var storageAccountName = 'st${workloadName}${take(resourceToken, 10)}'
var privateEndpointBlobName = 'pe-blob-${workloadName}-${environmentName}-${resourceToken}'
var privateEndpointFileName = 'pe-file-${workloadName}-${environmentName}-${resourceToken}'
var privateDnsZoneBlobName = 'privatelink.blob.${environment().suffixes.storage}'
var privateDnsZoneFileName = 'privatelink.file.${environment().suffixes.storage}'

// Container names
var lesionImagesContainer = 'lesion-images'
var chatFilesContainer = 'chat-files'
var backupContainer = 'backups'
var reportsContainer = 'reports'

// ========== STORAGE ACCOUNT ==========
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_GRS' // Geo-redundant storage for data protection
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    allowCrossTenantReplication: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    defaultToOAuthAuthentication: true
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
      virtualNetworkRules: []
      ipRules: []
    }
    publicNetworkAccess: 'Disabled'
    encryption: {
      services: {
        blob: {
          enabled: true
          keyType: 'Account'
        }
        file: {
          enabled: true
          keyType: 'Account'
        }
      }
      keySource: 'Microsoft.Storage'
      requireInfrastructureEncryption: true
    }
  }
}

// ========== BLOB CONTAINERS ==========
resource lesionImagesContainerResource 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: '${storageAccount.name}/default/${lesionImagesContainer}'
  properties: {
    publicAccess: 'None'
    metadata: {
      purpose: 'Storing patient lesion images for AI analysis'
    }
  }
}

resource chatFilesContainerResource 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: '${storageAccount.name}/default/${chatFilesContainer}'
  properties: {
    publicAccess: 'None'
    metadata: {
      purpose: 'Storing files shared in patient-doctor chat'
    }
  }
}

resource backupContainerResource 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: '${storageAccount.name}/default/${backupContainer}'
  properties: {
    publicAccess: 'None'
    metadata: {
      purpose: 'Storing database and application backups'
    }
  }
}

resource reportsContainerResource 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: '${storageAccount.name}/default/${reportsContainer}'
  properties: {
    publicAccess: 'None'
    metadata: {
      purpose: 'Storing ML model training and evaluation reports'
    }
  }
}

// ========== PRIVATE DNS ZONES ==========
resource privateDnsZoneBlob 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: privateDnsZoneBlobName
  location: 'global'
  tags: tags
}

resource privateDnsZoneFile 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: privateDnsZoneFileName
  location: 'global'
  tags: tags
}

// ========== PRIVATE DNS ZONE LINKS ==========
resource privateDnsZoneLinkBlob 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: privateDnsZoneBlob
  name: '${privateDnsZoneBlobName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnetId
    }
  }
}

resource privateDnsZoneLinkFile 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: privateDnsZoneFile
  name: '${privateDnsZoneFileName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnetId
    }
  }
}

// ========== PRIVATE ENDPOINTS ==========
resource privateEndpointBlob 'Microsoft.Network/privateEndpoints@2023-09-01' = {
  name: privateEndpointBlobName
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'blobConnection'
        properties: {
          privateLinkServiceId: storageAccount.id
          groupIds: [
            'blob'
          ]
        }
      }
    ]
  }
}

resource privateEndpointFile 'Microsoft.Network/privateEndpoints@2023-09-01' = {
  name: privateEndpointFileName
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'fileConnection'
        properties: {
          privateLinkServiceId: storageAccount.id
          groupIds: [
            'file'
          ]
        }
      }
    ]
  }
}

// ========== PRIVATE DNS ZONE GROUPS ==========
resource privateDnsZoneGroupBlob 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-09-01' = {
  parent: privateEndpointBlob
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'blob'
        properties: {
          privateDnsZoneId: privateDnsZoneBlob.id
        }
      }
    ]
  }
}

resource privateDnsZoneGroupFile 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-09-01' = {
  parent: privateEndpointFile
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'file'
        properties: {
          privateDnsZoneId: privateDnsZoneFile.id
        }
      }
    ]
  }
}

// ========== STORAGE LIFECYCLE MANAGEMENT ==========
resource lifecycleManagement 'Microsoft.Storage/storageAccounts/managementPolicies@2023-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    policy: {
      rules: [
        {
          name: 'DeleteOldBackups'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: {
              blobTypes: [
                'blockBlob'
              ]
              prefixMatch: [
                '${backupContainer}/'
              ]
            }
            actions: {
              baseBlob: {
                tierToCool: {
                  daysAfterModificationGreaterThan: 30
                }
                tierToArchive: {
                  daysAfterModificationGreaterThan: 90
                }
                delete: {
                  daysAfterModificationGreaterThan: 365
                }
              }
            }
          }
        }
        {
          name: 'ArchiveOldReports'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: {
              blobTypes: [
                'blockBlob'
              ]
              prefixMatch: [
                '${reportsContainer}/'
              ]
            }
            actions: {
              baseBlob: {
                tierToCool: {
                  daysAfterModificationGreaterThan: 60
                }
                tierToArchive: {
                  daysAfterModificationGreaterThan: 180
                }
              }
            }
          }
        }
      ]
    }
  }
}

// ========== OUTPUTS ==========
output storageAccountId string = storageAccount.id
output storageAccountName string = storageAccount.name
output storageConnectionString string = 'Will be configured via Key Vault'
output storageBlobEndpoint string = storageAccount.properties.primaryEndpoints.blob
output storageFileEndpoint string = storageAccount.properties.primaryEndpoints.file

// Container outputs
output lesionImagesContainer string = lesionImagesContainer
output chatFilesContainer string = chatFilesContainer
output backupContainer string = backupContainer
output reportsContainer string = reportsContainer
output blobContainerName string = lesionImagesContainer

// Private endpoint outputs
output privateEndpointBlobId string = privateEndpointBlob.id
output privateEndpointFileId string = privateEndpointFile.id
output privateDnsZoneBlobId string = privateDnsZoneBlob.id
output privateDnsZoneFileId string = privateDnsZoneFile.id