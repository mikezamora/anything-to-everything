/**
 * Audit API - Interface for audit log operations
 */

import { executePlugin } from './plugins';

export interface AuditLog {
  id: string;
  user_uuid: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  metadata?: string;
  ip_address?: string;
  user_agent?: string;
  created_at: number;
}

export interface AuditLogsResponse {
  logs: AuditLog[];
  total: number;
  pages: number;
}

/**
 * Get audit logs for a specific user with pagination
 */
export async function getUserAuditLogs(
  userUuid: string,
  page: number = 1,
  limit: number = 20
): Promise<AuditLogsResponse> {
  const result = await executePlugin<any, any>(
    'audit-plugin',
    'get_user_audit_logs',
    {
      user_uuid: userUuid,
      page,
      limit,
    }
  );

  if (!result.success || !result.data) {
    throw new Error(result.error || 'Failed to fetch audit logs');
  }

  return result.data;
}

/**
 * Get filtered audit logs
 */
export async function getFilteredAuditLogs(filters: {
  user_uuid?: string;
  action?: string;
  resource_type?: string;
  start_time?: number;
  end_time?: number;
  page?: number;
  limit?: number;
}): Promise<AuditLogsResponse> {
  const result = await executePlugin<any, any>(
    'audit-plugin',
    'get_audit_logs_filtered',
    filters
  );

  if (!result.success || !result.data) {
    throw new Error(result.error || 'Failed to fetch filtered audit logs');
  }

  return result.data;
}

/**
 * Create an audit log entry
 */
export async function createAuditLog(data: {
  user_uuid: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  metadata?: string;
  ip_address?: string;
  user_agent?: string;
}): Promise<void> {
  const result = await executePlugin<any, any>(
    'audit-plugin',
    'create_audit_log',
    data
  );

  if (!result.success) {
    throw new Error(result.error || 'Failed to create audit log');
  }
}
