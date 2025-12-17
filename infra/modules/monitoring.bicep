// Monitoring module - Application Insights, Log Analytics, and Azure Monitor
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

// ========== VARIABLES ==========
var logAnalyticsWorkspaceName = 'log-${workloadName}-${environmentName}-${resourceToken}'
var applicationInsightsName = 'appi-${workloadName}-${environmentName}-${resourceToken}'
var actionGroupName = 'ag-${workloadName}-${environmentName}-${resourceToken}'

// ========== LOG ANALYTICS WORKSPACE ==========
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsWorkspaceName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 90
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    workspaceCapping: {
      dailyQuotaGb: 1
    }
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// ========== APPLICATION INSIGHTS ==========
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// ========== ACTION GROUP ==========
resource actionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: actionGroupName
  location: 'global'
  tags: tags
  properties: {
    groupShortName: 'AiDoctorAG'
    enabled: true
    emailReceivers: []
    smsReceivers: []
    webhookReceivers: []
    azureAppPushReceivers: []
    armRoleReceivers: [
      {
        name: 'Owner'
        roleId: '8e3af657-a8ff-443c-a75c-2fe8c4bcb635'
        useCommonAlertSchema: true
      }
    ]
  }
}

// ========== METRIC ALERTS ==========

// High CPU alert
resource highCpuAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'High CPU Usage - AI Doctor'
  location: 'global'
  tags: tags
  properties: {
    description: 'Alert when CPU usage is high'
    severity: 2
    enabled: true
    scopes: []
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'HighCPU'
          metricName: 'Percentage CPU'
          operator: 'GreaterThan'
          threshold: 80
          timeAggregation: 'Average'
          criterionType: 'StaticThresholdCriterion'
        }
      ]
    }
    actions: [
      {
        actionGroupId: actionGroup.id
      }
    ]
  }
}

// High memory alert
resource highMemoryAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'High Memory Usage - AI Doctor'
  location: 'global'
  tags: tags
  properties: {
    description: 'Alert when memory usage is high'
    severity: 2
    enabled: true
    scopes: []
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'HighMemory'
          metricName: 'MemoryPercentage'
          operator: 'GreaterThan'
          threshold: 85
          timeAggregation: 'Average'
          criterionType: 'StaticThresholdCriterion'
        }
      ]
    }
    actions: [
      {
        actionGroupId: actionGroup.id
      }
    ]
  }
}

// High response time alert
resource highResponseTimeAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'High Response Time - AI Doctor'
  location: 'global'
  tags: tags
  properties: {
    description: 'Alert when response time is high'
    severity: 2
    enabled: true
    scopes: []
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'HighResponseTime'
          metricName: 'AverageResponseTime'
          operator: 'GreaterThan'
          threshold: 5000
          timeAggregation: 'Average'
          criterionType: 'StaticThresholdCriterion'
        }
      ]
    }
    actions: [
      {
        actionGroupId: actionGroup.id
      }
    ]
  }
}

// Failed requests alert
resource failedRequestsAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'Failed Requests - AI Doctor'
  location: 'global'
  tags: tags
  properties: {
    description: 'Alert when there are failed requests'
    severity: 1
    enabled: true
    scopes: []
    evaluationFrequency: 'PT1M'
    windowSize: 'PT5M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'FailedRequests'
          metricName: 'Http5xx'
          operator: 'GreaterThan'
          threshold: 5
          timeAggregation: 'Total'
          criterionType: 'StaticThresholdCriterion'
        }
      ]
    }
    actions: [
      {
        actionGroupId: actionGroup.id
      }
    ]
  }
}

// ========== WORKBOOK ==========
resource workbook 'Microsoft.Insights/workbooks@2023-06-01' = {
  name: guid(resourceGroup().id, 'ai-doctor-workbook')
  location: location
  tags: tags
  kind: 'shared'
  properties: {
    displayName: 'AI Doctor Application Dashboard'
    serializedData: string({
      version: 'Notebook/1.0'
      items: [
        {
          type: 1
          content: {
            json: '# AI Doctor Application Monitoring Dashboard\\n\\nThis dashboard provides comprehensive monitoring for the AI Doctor application including performance metrics, error rates, and user analytics.'
          }
        }
        {
          type: 3
          content: {
            version: 'KqlItem/1.0'
            query: 'requests\\n| where timestamp > ago(24h)\\n| summarize Count = count() by bin(timestamp, 1h)\\n| render timechart'
            size: 0
            title: 'Request Volume (24h)'
            queryType: 0
            resourceType: 'microsoft.insights/components'
          }
        }
        {
          type: 3
          content: {
            version: 'KqlItem/1.0'
            query: 'requests\\n| where timestamp > ago(24h)\\n| summarize AverageResponseTime = avg(duration) by bin(timestamp, 1h)\\n| render timechart'
            size: 0
            title: 'Average Response Time (24h)'
            queryType: 0
            resourceType: 'microsoft.insights/components'
          }
        }
        {
          type: 3
          content: {
            version: 'KqlItem/1.0'
            query: 'exceptions\\n| where timestamp > ago(24h)\\n| summarize Count = count() by bin(timestamp, 1h), type\\n| render timechart'
            size: 0
            title: 'Exception Count by Type (24h)'
            queryType: 0
            resourceType: 'microsoft.insights/components'
          }
        }
      ]
    })
    category: 'workbook'
    sourceId: applicationInsights.id
  }
}

// ========== OUTPUTS ==========
output logAnalyticsWorkspaceId string = logAnalyticsWorkspace.id
output logAnalyticsWorkspaceName string = logAnalyticsWorkspace.name
output logAnalyticsCustomerId string = logAnalyticsWorkspace.properties.customerId

output applicationInsightsId string = applicationInsights.id
output applicationInsightsName string = applicationInsights.name
output applicationInsightsInstrumentationKey string = applicationInsights.properties.InstrumentationKey
output applicationInsightsConnectionString string = applicationInsights.properties.ConnectionString

output actionGroupId string = actionGroup.id
output actionGroupName string = actionGroup.name

output workbookId string = workbook.id